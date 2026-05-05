"""
MolecularDataset with MultiTask Support
Passes both classification (y_cls) and regression (y_active, pEC50) labels
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem
import numpy as np
from mendeleev import element
from molvs import Standardizer
import torch
from torch.utils.data import TensorDataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import logging
import os

logger = logging.getLogger(__name__)
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)


_WORKER_DATASET = None


def _init_molecular_dataset_worker(node_block, pEC50_mean, pEC50_std):
    global _WORKER_DATASET
    helper = MolecularDataset.__new__(MolecularDataset)
    helper.node_block = node_block
    helper.global_dim = 0
    helper.num_node_features = 0
    helper.edge_dim = 0
    helper._mendeleev_cache = {}
    helper.standardizer = Standardizer()
    helper.pEC50_mean = pEC50_mean
    helper.pEC50_std = pEC50_std
    helper.hybridization_dict = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 0.5,
        Chem.rdchem.HybridizationType.SP3: 1,
    }
    _WORKER_DATASET = helper


def _process_molecule_worker(payload):
    smiles, name, label, pEC50 = payload
    if _WORKER_DATASET is None:
        raise RuntimeError('Worker dataset is not initialized')

    data, mol, debug_info = _WORKER_DATASET.smiles_to_data(
        smiles,
        name,
        label,
        pEC50,
        return_mol=True,
        return_debug=True,
    )

    if data is None or mol is None:
        return {
            'ok': False,
            'name': name,
            'smiles': smiles,
            'label': label,
            'pEC50': pEC50,
            'debug': debug_info,
        }

    return {
        'ok': True,
        'name': name,
        'smiles': smiles,
        'label': label,
        'pEC50': pEC50,
        'data_payload': {
            'x': data.x.detach().cpu().tolist(),
            'edge_index': data.edge_index.detach().cpu().tolist(),
            'edge_attr': data.edge_attr.detach().cpu().tolist(),
            'u': data.u.detach().cpu().tolist(),
        },
        'molblock': Chem.MolToMolBlock(mol),
        'debug': debug_info,
    }


class MolecularDataset:
    def __init__(self, smiles_list, names_list, labels=None, pEC50_labels=None,
                 activity_threshold=5.0, node_block="BMP", source_file=None):
        """
        Args:
            smiles_list: List of SMILES strings
            names_list: List of compound names
            labels: Binary classification labels (0/1 for active/inactive)
            pEC50_labels: Regression labels (pEC50 values, can be None for some compounds)
            activity_threshold: pEC50 threshold for classification (e.g., 5.0 means pEC50 > 5 = active)
            node_block: Type of GNN block to use
        """
        self.smiles_list = smiles_list.copy()
        self.names_list = names_list.copy()
        self.labels = labels.copy() if labels is not None else [None] * len(smiles_list)
        self.pEC50_labels = pEC50_labels.copy() if pEC50_labels is not None else [None] * len(smiles_list)
        self.activity_threshold = activity_threshold
        self.data_list = []
        self.node_block = node_block
        self.global_dim = 0
        self.num_node_features = 0
        self.edge_dim = 0
        self.successful_labels = []
        self.successful_pEC50 = []
        self.successful_names = []
        self.successful_smiles = []
        self._mendeleev_cache = {}
        self.processed_count = 0
        self.standardizer = Standardizer()
        self.source_file = source_file
        self.preprocess_workers = max(1, int(os.getenv("BMPNN_PREPROCESS_WORKERS", "1")))
        self.sdf_path = self._resolve_sdf_path(source_file)
        self._sdf_records = self._load_sdf_records(self.sdf_path) if self.sdf_path and self.sdf_path.exists() else {}
        self._sdf_dirty = False

        # Statistics for normalization
        self.pEC50_mean = None
        self.pEC50_std = None

        self.hybridization_dict = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 0.5,
            Chem.rdchem.HybridizationType.SP3: 1,
        }

        print(f"Number of molecules in dataset: {len(smiles_list)}")
        print(f"Using Model: {self.node_block}")

        # Compute pEC50 statistics for normalization, ignoring missing/non-finite values
        valid_pEC50 = [
            float(x) for x in self.pEC50_labels
            if x is not None and np.isfinite(x)
        ]
        if len(valid_pEC50) > 0:
            self.pEC50_mean = np.mean(valid_pEC50)
            self.pEC50_std = np.std(valid_pEC50)
            # Ensure std is not too small to avoid division issues
            if self.pEC50_std < 0.1:
                self.pEC50_std = 1.0
            print(f"pEC50 statistics: mean={self.pEC50_mean:.3f}, std={self.pEC50_std:.3f}")
            print(f"Compounds with pEC50 labels: {len(valid_pEC50)}/{len(smiles_list)}")

        logger.info("Converting SMILES to data objects.")
        debug_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "undefined_stereo": 0,
            "total_enumerated_isomers": 0,
            "total_successful_conformers": 0,
            "molecules_from_smiles": 0,
        }
        valid_smiles = []
        valid_names = []
        valid_labels = []
        valid_pEC50 = []
        pending = []

        for smiles, name, label, pEC50 in zip(self.smiles_list, self.names_list, self.labels, self.pEC50_labels):
            cache_key = self._make_cache_key(name, smiles)

            try:
                if cache_key in self._sdf_records:
                    cached_mol = self._sdf_records[cache_key]
                    data, label, pEC50 = self.data_from_sdf(cached_mol, name, smiles, label, pEC50)
                    needs_cache_refresh = (
                        (label is not None and not cached_mol.HasProp("label"))
                        or (pEC50 is not None and not cached_mol.HasProp("pEC50"))
                    )
                    if needs_cache_refresh:
                        self._sdf_records[cache_key] = self._build_sdf_mol(
                            cached_mol, data, name, smiles, label=label, pEC50=pEC50
                        )
                        self._sdf_dirty = True

                    if data is not None:
                        self.data_list.append(data)
                        self.successful_labels.append(label)
                        self.successful_pEC50.append(pEC50)
                        self.successful_names.append(name)
                        self.successful_smiles.append(smiles)
                        valid_smiles.append(smiles)
                        valid_names.append(name)
                        valid_labels.append(label)
                        valid_pEC50.append(pEC50)
                        self.processed_count += 1
                        debug_stats["cache_hits"] += 1
                    else:
                        logger.warning(f"Skipping invalid cached molecule: Name: {name}, SMILES: {smiles}")
                else:
                    pending.append((smiles, name, label, pEC50))
                    debug_stats["cache_misses"] += 1
            except Exception as e:
                logger.error(f"Failed to process cached molecule: Name: {name}, SMILES: {smiles}, Error: {e}")

        if pending:
            worker_count = min(self.preprocess_workers, len(pending))
            logger.info(
                f"Processing {len(pending)} uncached molecules with {worker_count} worker(s)"
            )

            if worker_count > 1:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_molecular_dataset_worker,
                    initargs=(self.node_block, self.pEC50_mean, self.pEC50_std),
                ) as executor:
                    results_iter = executor.map(_process_molecule_worker, pending)
                    results = list(results_iter)
            else:
                _init_molecular_dataset_worker(self.node_block, self.pEC50_mean, self.pEC50_std)
                results = [_process_molecule_worker(item) for item in pending]

            for result in results:
                debug_info = result.get("debug") or {}
                debug_stats["molecules_from_smiles"] += 1
                debug_stats["undefined_stereo"] += int(bool(debug_info.get("undefined_stereo", False)))
                debug_stats["total_enumerated_isomers"] += int(debug_info.get("enumerated_isomers", 0))
                debug_stats["total_successful_conformers"] += int(debug_info.get("successful_conformers", 0))

                if not result.get("ok"):
                    logger.warning(
                        f"Skipping invalid molecule: Name: {result.get('name')}, SMILES: {result.get('smiles')}"
                    )
                    continue

                label = result["label"]
                pEC50 = result["pEC50"]
                name = result["name"]
                smiles = result["smiles"]
                payload = result["data_payload"]
                data = self._build_data_object(
                    atom_features=torch.tensor(payload["x"], dtype=torch.float),
                    edge_index=torch.tensor(payload["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(payload["edge_attr"], dtype=torch.float),
                    global_features=torch.tensor(payload["u"], dtype=torch.float),
                    name=name,
                    smiles=smiles,
                    label=label,
                    pEC50=pEC50,
                )

                mol = Chem.MolFromMolBlock(result["molblock"], removeHs=False)
                if data is not None and mol is not None and self.sdf_path is not None:
                    cache_key = self._make_cache_key(name, smiles)
                    self._sdf_records[cache_key] = self._build_sdf_mol(
                        mol, data, name, smiles, label=label, pEC50=pEC50
                    )
                    self._sdf_dirty = True

                self.data_list.append(data)
                self.successful_labels.append(label)
                self.successful_pEC50.append(pEC50)
                self.successful_names.append(name)
                self.successful_smiles.append(smiles)
                valid_smiles.append(smiles)
                valid_names.append(name)
                valid_labels.append(label)
                valid_pEC50.append(pEC50)
                self.processed_count += 1

        self.smiles_list = valid_smiles
        self.names_list = valid_names
        self.labels = valid_labels
        self.pEC50_labels = valid_pEC50

        if self._sdf_dirty and self.sdf_path is not None:
            self._write_sdf_records()

        processed_uncached = debug_stats["molecules_from_smiles"]
        if processed_uncached > 0:
            avg_isomers = debug_stats["total_enumerated_isomers"] / processed_uncached
            avg_conformers = debug_stats["total_successful_conformers"] / processed_uncached
            logger.info(
                "Preprocessing profile | uncached=%d undefined_stereo=%d avg_enumerated_isomers=%.2f avg_successful_conformers=%.2f",
                processed_uncached,
                debug_stats["undefined_stereo"],
                avg_isomers,
                avg_conformers,
            )
            print(
                f"Preprocessing profile: uncached={processed_uncached}, "
                f"undefined_stereo={debug_stats['undefined_stereo']}, "
                f"avg_enumerated_isomers={avg_isomers:.2f}, "
                f"avg_successful_conformers={avg_conformers:.2f}"
            )

        logger.info(f"Processed {self.processed_count} valid molecules out of {len(smiles_list)} provided.")

    def _delete_at_index(self, i):
        """Helper to delete items at index across all lists"""
        del self.smiles_list[i]
        del self.names_list[i]
        if i < len(self.labels):
            del self.labels[i]
        if i < len(self.pEC50_labels):
            del self.pEC50_labels[i]

    def _resolve_sdf_path(self, source_file):
        if not source_file:
            return None
        path = Path(source_file).expanduser().resolve()
        if path.suffix.lower() == ".sdf":
            return path
        return path.with_suffix(".sdf")

    def _make_cache_key(self, name, smiles):
        clean_name = str(name).replace("\r", " ").replace("\n", " ").strip()
        clean_smiles = str(smiles).replace("\r", " ").replace("\n", " ").strip()
        return f"{clean_name}||{clean_smiles}"

    def _load_sdf_records(self, sdf_path):
        records = {}
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            if mol.HasProp("_cache_key"):
                records[mol.GetProp("_cache_key")] = mol
        logger.info(f"Loaded {len(records)} cached molecules from {sdf_path}")
        return records

    def _write_sdf_records(self):
        self.sdf_path.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(self.sdf_path))
        for cache_key in sorted(self._sdf_records):
            writer.write(self._sdf_records[cache_key])
        writer.close()
        logger.info(f"Wrote {len(self._sdf_records)} cached molecules to {self.sdf_path}")

    def _build_data_object(self, atom_features, edge_index, edge_attr, global_features, name, smiles, label=None, pEC50=None):
        self.num_node_features = atom_features.size(1)
        self.edge_dim = edge_attr.size(1) if edge_attr.ndim > 1 and edge_attr.numel() > 0 else 0
        self.global_dim = global_features.size(1) if global_features.ndim > 1 else global_features.numel()

        if label is not None:
            y_cls = torch.tensor([label], dtype=torch.float).reshape(-1, 1)
        else:
            y_cls = torch.zeros(1, 1)

        if pEC50 is not None and np.isfinite(pEC50):
            if self.pEC50_mean is not None and self.pEC50_std is not None:
                normalized_pEC50 = (pEC50 - self.pEC50_mean) / (self.pEC50_std + 1e-8)
            else:
                normalized_pEC50 = pEC50
            y_active = torch.tensor([normalized_pEC50], dtype=torch.float).reshape(-1, 1)
            mask_active = torch.tensor([1.0], dtype=torch.float).reshape(-1, 1)
        else:
            y_active = torch.zeros(1, 1)
            mask_active = torch.tensor([0.0], dtype=torch.float).reshape(-1, 1)

        data = Data(
            x=atom_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=global_features,
            y=y_cls,
            y_cls=y_cls,
            y_active=y_active,
            mask_active=mask_active
        )
        data.smiles = smiles
        data.name = name
        return data

    def _coerce_optional_label(self, value):
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return int(numeric_value)

    def _coerce_optional_pEC50(self, value):
        if value is None:
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric_value):
            return None
        return numeric_value

    def _get_optional_sdf_prop(self, mol, key, caster):
        if not mol.HasProp(key):
            return None
        try:
            return caster(mol.GetProp(key))
        except (TypeError, ValueError):
            return None

    def _build_sdf_mol(self, mol, data, name, smiles, label=None, pEC50=None):
        clean_name = str(name).replace("\r", " ").replace("\n", " ").strip()
        clean_smiles = str(smiles).replace("\r", " ").replace("\n", " ").strip()
        sdf_mol = Chem.Mol(mol)
        sdf_mol.SetProp("_Name", clean_name)
        sdf_mol.SetProp("_cache_key", self._make_cache_key(clean_name, clean_smiles))
        sdf_mol.SetProp("smiles", clean_smiles)
        sdf_mol.SetProp("node_block", self.node_block)
        sdf_mol.SetProp("x", json.dumps(data.x.detach().cpu().tolist()))
        sdf_mol.SetProp("edge_index", json.dumps(data.edge_index.detach().cpu().tolist()))
        sdf_mol.SetProp("edge_attr", json.dumps(data.edge_attr.detach().cpu().tolist()))
        sdf_mol.SetProp("u", json.dumps(data.u.detach().cpu().tolist()))

        label = self._coerce_optional_label(label)
        pEC50 = self._coerce_optional_pEC50(pEC50)
        if label is not None:
            sdf_mol.SetIntProp("label", int(label))
        if pEC50 is not None:
            sdf_mol.SetDoubleProp("pEC50", float(pEC50))
        return sdf_mol

    def data_from_sdf(self, mol, name, smiles, label=None, pEC50=None):
        atom_features = torch.tensor(json.loads(mol.GetProp("x")), dtype=torch.float)
        edge_index = torch.tensor(json.loads(mol.GetProp("edge_index")), dtype=torch.long)
        edge_attr = torch.tensor(json.loads(mol.GetProp("edge_attr")), dtype=torch.float)
        global_features = torch.tensor(json.loads(mol.GetProp("u")), dtype=torch.float)

        label = self._coerce_optional_label(label)
        pEC50 = self._coerce_optional_pEC50(pEC50)
        if label is None:
            label = self._get_optional_sdf_prop(mol, "label", lambda v: int(float(v)))
        if pEC50 is None:
            pEC50 = self._get_optional_sdf_prop(mol, "pEC50", float)

        data = self._build_data_object(
            atom_features=atom_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            name=name,
            smiles=smiles,
            label=label,
            pEC50=pEC50,
        )
        return data, label, pEC50


    def smiles_to_data(self, smiles, name, label=None, pEC50=None, output_dir="molecule_images", return_mol=False, return_debug=False):

        debug_info = {
            "undefined_stereo": False,
            "enumerated_isomers": 0,
            "successful_conformers": 0,
        }

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                if return_mol:
                    return (None, None, debug_info) if return_debug else (None, None)
                return (None, debug_info) if return_debug else None

            mol = self.standardizer.standardize(mol)
            mol = self.correct_atom_types(mol)
            if mol is None:
                logger.warning(f"Failed after standardization/correction: {smiles}")
                if return_mol:
                    return (None, None, debug_info) if return_debug else (None, None)
                return (None, debug_info) if return_debug else None

            Chem.AssignStereochemistry(mol, force=False, cleanIt=True)

            # -----------------------------
            # 1) Check whether stereo is fully defined
            # -----------------------------
            chiral_centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, useLegacyImplementation=False
            )
            has_undefined_atom_stereo = any(tag == "?" for _, tag in chiral_centers)

            # Optional: also consider undefined double-bond stereo if relevant
            # For many workflows, atom stereocenters are the main issue.

            # -----------------------------
            # 2) Enumerate only when needed
            # -----------------------------
            if has_undefined_atom_stereo:
                opts = StereoEnumerationOptions(
                    tryEmbedding=True,
                    unique=True,
                    onlyUnassigned=True,
                    maxIsomers=24,   # adjust if needed
                )
                candidate_mols = list(EnumerateStereoisomers(mol, options=opts))
                if not candidate_mols:
                    candidate_mols = [mol]
            else:
                candidate_mols = [mol]

            debug_info["undefined_stereo"] = has_undefined_atom_stereo
            debug_info["enumerated_isomers"] = len(candidate_mols)

            best_mol = None
            best_conf = None
            best_energy = np.inf

            # -----------------------------
            # 3) For each candidate stereoisomer:
            #    embed multiple conformers, optimize, keep best
            # -----------------------------
            for candidate in candidate_mols:
                try:
                    cand = Chem.Mol(candidate)
                    Chem.AssignStereochemistry(cand, force=False, cleanIt=True)

                    cand_h = Chem.AddHs(cand)

                    params = AllChem.ETKDGv3()
                    params.enforceChirality = True
                    params.useSmallRingTorsions = True
                    params.useBasicKnowledge = True

                    conf_ids = AllChem.EmbedMultipleConfs(
                        cand_h,
                        numConfs=5,   # adjust
                        params=params
                    )

                    if not conf_ids:
                        continue

                    mp = AllChem.MMFFGetMoleculeProperties(cand_h)
                    if mp is None:
                        continue

                    local_best_cid = None
                    local_best_energy = np.inf
                    successful_conf_count = 0

                    for cid in conf_ids:
                        try:
                            ff = AllChem.MMFFGetMoleculeForceField(cand_h, mp, confId=cid)
                            if ff is None:
                                continue
                            ff.Minimize()
                            e = ff.CalcEnergy()

                            successful_conf_count += 1
                            if e < local_best_energy:
                                local_best_energy = e
                                local_best_cid = cid
                        except Exception:
                            continue

                    debug_info["successful_conformers"] += successful_conf_count

                    if local_best_cid is None:
                        continue

                    # Keep global best stereoisomer+conformer
                    if local_best_energy < best_energy:
                        best_energy = local_best_energy

                        # assign stereo from 3D but do NOT overwrite existing tags
                        Chem.AssignStereochemistryFrom3D(
                            cand_h,
                            confId=local_best_cid,
                            replaceExistingTags=False
                        )

                        # remove Hs and transfer the chosen conformer
                        cand_no_h = Chem.RemoveHs(cand_h)

                        conf_h = cand_h.GetConformer(local_best_cid)
                        conf = Chem.Conformer(cand_no_h.GetNumAtoms())
                        for atom_id in range(cand_no_h.GetNumAtoms()):
                            pos = conf_h.GetAtomPosition(atom_id)
                            conf.SetAtomPosition(atom_id, pos)

                        best_mol = cand_no_h
                        best_conf = conf

                except Exception as e:
                    logger.warning(f"Candidate stereoisomer failed for {smiles}: {e}")
                    continue

            if best_mol is None or best_conf is None:
                logger.error(f"No valid stereoisomer/conformer generated for: {smiles}")
                if return_mol:
                    return (None, None, debug_info) if return_debug else (None, None)
                return (None, debug_info) if return_debug else None

            best_mol.RemoveAllConformers()
            best_mol.AddConformer(best_conf)
            data = self.extract_features(best_mol, best_conf, name, smiles, label, pEC50, output_dir)
            if return_mol:
                return (data, best_mol, debug_info) if return_debug else (data, best_mol)
            return (data, debug_info) if return_debug else data

        except Exception as e:
            logger.error(f"General failure processing SMILES: {smiles}, Error: {e}")
            if return_mol:
                return (None, None, debug_info) if return_debug else (None, None)
            return (None, debug_info) if return_debug else None

    def extract_features(self, mol, conf, name, smiles, label=None, pEC50=None, output_dir="molecule_images"):
        try:
            atom_features = self.get_atom_features(mol, conf)
            edge_index, edge_attr = self.get_edge_index_and_features(mol, conf, self.node_block)
            if edge_index is None or edge_attr is None:
                return None
            if edge_index.numel() > 0 and edge_index.max().item() >= atom_features.size(0):
                logger.error(f"Invalid edge index detected: {edge_index.max().item()} exceeds number of atoms {atom_features.size(0)}")
                return None
            global_features = self.get_global_features(mol, conf)
            return self._build_data_object(
                atom_features=atom_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_features=global_features,
                name=name,
                smiles=smiles,
                label=label,
                pEC50=pEC50,
            )

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def get_global_features(self, mol, conf):
        try:
            chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))/6
            hba = abs(1/(1e-5 + 10 * (Descriptors.NumHDonors(mol) / 5) + abs(Descriptors.NumHAcceptors(mol) / 10)))
            rotatable = Descriptors.NumRotatableBonds(mol)/10
            tpsa_logp = (Descriptors.TPSA(mol) + Descriptors.MolLogP(mol))/145
            fsp3 = Descriptors.FractionCSP3(mol)
            rog = self.calculate_radius_of_gyration(mol, conf)/5

            global_features = [chiral, hba, rotatable, tpsa_logp, fsp3, rog]

            # Replace any NaN/Inf values with 0
            global_features = [0.0 if (f != f or abs(f) == float('inf')) else f for f in global_features]

        except Exception:
            global_features = [0.0] * 6

        self.global_dim = len(global_features)
        return torch.tensor(global_features, dtype=torch.float).unsqueeze(0)

    def calculate_radius_of_gyration(self, mol, conf):
        try:
            coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
            total_mass = np.sum(masses)
            center_of_mass = np.sum(coords.T * masses, axis=1) / total_mass
            rg_square = np.sum(masses * np.sum((coords - center_of_mass) ** 2, axis=1)) / total_mass
            return np.sqrt(rg_square)
        except:
            return 0.0

    def get_cached_element_props(self, atomic_num):
        if atomic_num not in self._mendeleev_cache:
            el = element(atomic_num)
            self._mendeleev_cache[atomic_num] = {
                "electronegativity": (el.electronegativity('pauling') - 0.9) / 3.1,
                "polarizability": (el.dipole_polarizability - 4.5) / (35 - 4.5),
                "vdw_radius": (el.vdw_radius - 120) / (166 - 120)
            }
        return self._mendeleev_cache[atomic_num]

    def calculate_buried_volume(self, mol, conf, atom_idx, radius=3.5, grid_spacing=0.75):
        central_atom_pos = conf.GetAtomPosition(atom_idx)
        central_point = np.array([central_atom_pos.x, central_atom_pos.y, central_atom_pos.z])
        grid = np.arange(-radius, radius + grid_spacing, grid_spacing)
        grid_points = np.array(np.meshgrid(grid, grid, grid)).reshape(3, -1).T
        sphere_mask = np.linalg.norm(grid_points, axis=1) <= radius
        sphere_points = grid_points[sphere_mask] + central_point
        occupied_count = 0
        for i, atom in enumerate(mol.GetAtoms()):
            if i == atom_idx:
                continue
            atom_pos = conf.GetAtomPosition(i)
            atom_pos_array = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
            distance = np.linalg.norm(sphere_points - atom_pos_array, axis=1)
            vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
            occupied_count += np.sum(distance <= vdw_radius)
        total_points = len(sphere_points)
        return occupied_count / total_points

    def correct_atom_types(self, mol):
        corrections = {
            "Cu+2": 29, "Se+2": 34, "Rh+6": 45, "W+6": 74, "Co+3": 27,
            "Zn+2": 30, "Ni+2": 28, "Pd+2": 46, "Gd+3": 64, "Re+5": 75,
            "Pt+2": 78, "Cr3+3": 24, "Zr2": 40, "Ba": 56, "Ti+4": 22,
        }
        for atom in mol.GetAtoms():
            formal_charge = atom.GetFormalCharge()
            symbol = atom.GetSymbol()
            charge_sign = "+" if formal_charge >= 0 else ""
            key = f"{symbol}{charge_sign}{formal_charge}"
            if key in corrections:
                atomic_num = corrections[key]
                atom.SetAtomicNum(atomic_num)
                atom.SetFormalCharge(formal_charge)
        return mol

    def get_atom_features(self, mol, conf):
        """Get atom features for all atoms INCLUDING hydrogens.
        This preserves stereochemistry information from explicit H atoms.
        """
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            props = self.get_cached_element_props(atomic_num)
            try:
                bv = self.calculate_buried_volume(mol, conf, atom.GetIdx())
                if bv != bv or abs(bv) == float('inf'):
                    bv = 0.5
            except:
                bv = 0.5
            atom_feature = [
                (atomic_num - 1) / 178,
                bv,
                self.hybridization_dict.get(atom.GetHybridization(), 0),
                props["electronegativity"],
                props["polarizability"],
                props["vdw_radius"]
            ]

            # Handle any NaN/Inf
            atom_feature = [0.0 if (f != f or abs(f) == float('inf')) else f for f in atom_feature]
            atom_features.append(atom_feature)

        # Handle empty molecules
        if len(atom_features) == 0:
            atom_features = [[0.0] * 6]
            logger.warning("Molecule has no atoms")

        self.num_node_features = len(atom_feature)
        return torch.tensor(atom_features, dtype=torch.float)

    def get_ring_size_feature(self, bond):
        if not bond.IsInRing():
            return 0.0
        elif bond.IsInRingSize(3):
            return 0.14
        elif bond.IsInRingSize(4):
            return 0.28
        elif bond.IsInRingSize(5):
            return 0.42
        elif bond.IsInRingSize(6):
            return 0.57
        elif bond.IsInRingSize(7):
            return 0.71
        elif bond.IsInRingSize(8):
            return 0.85
        return 1.0

    def get_edge_index_and_features(self, mol, conf, node_block):
        edge_index = []
        edge_attr = []
        try:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i >= mol.GetNumAtoms() or j >= mol.GetNumAtoms():
                    continue
                bond_length = Chem.rdMolTransforms.GetBondLength(conf, i, j)
                edge_feature = [
                    (bond_length - 1.05161541)/(2.4620574 - 1.05161541),
                    bond.GetBondTypeAsDouble()/2,
                    1 if bond.GetIsConjugated() else 0,
                    self.get_ring_size_feature(bond)
                ]
                if node_block == "UMP":
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append(edge_feature)
                    edge_attr.append(edge_feature)
                else:
                    edge_index.append([i, j])
                    edge_attr.append(edge_feature)
        except Exception as e:
            print(f"Error processing bond features: {e}")
            return None, None
        self.edge_dim = len(edge_feature)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return edge_index, edge_attr

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, but got {type(idx)}")
        return self.data_list[idx]
