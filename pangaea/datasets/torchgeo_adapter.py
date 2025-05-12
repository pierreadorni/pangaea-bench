import torch
from torch.utils.data import Subset, random_split
import torchgeo.datasets
import logging
import os
import re

from pangaea.datasets.base import RawGeoFMDataset

band_to_modality = {
    "optical": [
        re.compile(r"^B\d{2}$", re.IGNORECASE),  # e.g., B01, B12
        re.compile(r"^B\d{1}$", re.IGNORECASE),  # e.g., B1, B2
        re.compile(r"^B\d{3}$", re.IGNORECASE),  # e.g., B101, B200
        re.compile(r"^red$", re.IGNORECASE),
        re.compile(r"^green$", re.IGNORECASE),
        re.compile(r"^blue$", re.IGNORECASE),
        re.compile(r"^nir$", re.IGNORECASE),
        re.compile(r"^swir\d?$", re.IGNORECASE), #swir, swir1, swir2
    ],
    "sar": [
        re.compile(r"^VH$", re.IGNORECASE),
        re.compile(r"^VV$", re.IGNORECASE),
        re.compile(r"^HH$", re.IGNORECASE),
        re.compile(r"^HV$", re.IGNORECASE),
    ],
}

class TorchGeoAdapter(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        root_path: str,
        classes: list | None = None, # Renamed from 'classes'
        num_classes: int | None = None, # Renamed from 'num_classes'
        ignore_index: int | None = None, # Renamed from 'ignore_index'
        img_size: int = 224,
        bands: dict[str, list[str]] | None = None, # Renamed from 'bands'
        distribution: list[int] | None = None, # Renamed from 'distribution'
        data_mean: dict[str, list[float]] | None = None, # Renamed from 'data_mean'
        data_std: dict[str, list[float]] | None = None, # Renamed from 'data_std'
        data_min: dict[str, list[float]] | None = None, # Renamed from 'data_min'
        data_max: dict[str, list[float]] | None = None, # Renamed from 'data_max'
        download_url: str | None = None,
        auto_download: bool = False,
        multi_modal: bool = False, # New: whether to use multi-modal data
        multi_temporal: bool | int = False, # New: whether to use multi-temporal data
        torchgeo_dataset_name: str = "Landsat8",
        torchgeo_dataset_kwargs: dict | None = None,
        torchgeo_val_ratio: float | None = None, # New: ratio for val split from train
        cfg_seed: int | None = None, # New: seed for reproducible splits
        **kwargs # To catch any other RawGeoFMDataset args
    ):

        self.torchgeo_dataset_class = getattr(torchgeo.datasets, torchgeo_dataset_name)

        # Initialize RawGeoFMDataset with values from config (some might be placeholders)
        super(TorchGeoAdapter, self).__init__(
            split=split,
            dataset_name=dataset_name,
            # multi_modal/multi_temporal will be determined or use defaults from base but we need to give them anyway
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes if classes is not None else [],
            num_classes=num_classes if num_classes is not None else 0,
            ignore_index=ignore_index if ignore_index is not None else -100,
            img_size=img_size,
            bands=bands if bands is not None else {},
            distribution=distribution,
            data_mean=data_mean if data_mean is not None else {},
            data_std=data_std if data_std is not None else {},
            data_min=data_min if data_min is not None else {},
            data_max=data_max if data_max is not None else {},
            download_url=download_url,
            auto_download=auto_download, # This refers to RawGeoFMDataset's download, not torchgeo's
        )

        self.torchgeo_val_ratio = torchgeo_val_ratio
        self.cfg_seed = cfg_seed

        
        _torchgeo_dataset_kwargs = dict(torchgeo_dataset_kwargs) if torchgeo_dataset_kwargs is not None else {}

        # --- Step 1: Instantiate a reference torchgeo dataset to get metadata ---
        # This instance uses a canonical split (e.g., 'train') or the user-provided split in kwargs.
        ref_kwargs = _torchgeo_dataset_kwargs.copy()
        # Default to 'train' for metadata, but allow override via torchgeo_dataset_kwargs.split
        ref_split_for_metadata = ref_kwargs.pop('split', 'train')

        # Ensure root path exists for torchgeo if auto_download is True for torchgeo
        # Pangaea's RawGeoFMDataset might also create it via its own download logic if its auto_download is True.
        if auto_download and not os.path.exists(self.root_path):
            os.makedirs(self.root_path, exist_ok=True)
        
        try:
            reference_torchgeo_dataset = self.torchgeo_dataset_class(
                root=self.root_path,
                split=ref_split_for_metadata,
                download=auto_download, # Let torchgeo handle its download for this ref instance
                **ref_kwargs
            )
        except Exception as e:
            print(f"Error instantiating reference torchgeo dataset {torchgeo_dataset_name} with split '{ref_split_for_metadata}': {e}")
            print(f"Make sure the split '{ref_split_for_metadata}' is valid for {torchgeo_dataset_name} or provide a valid 'split' in 'torchgeo_dataset_kwargs'.")
            raise
  
        # --- Step 2: Derive/Update metadata from reference_torchgeo_dataset ---
        # Update self.classes (List of class names)
        derived_classes = None
        if hasattr(reference_torchgeo_dataset, 'classes') and reference_torchgeo_dataset.classes:
            if torchgeo_dataset_name in ["Vaihingen2D", "Potsdam2D"] and \
               isinstance(reference_torchgeo_dataset.classes, list) and \
               len(reference_torchgeo_dataset.classes) > 0 and \
               isinstance(reference_torchgeo_dataset.classes[0], dict) and \
               'name' in reference_torchgeo_dataset.classes[0]:
                try:
                    derived_classes = [c['name'] for c in reference_torchgeo_dataset.classes]
                except (TypeError, KeyError): # Fallback if structure is unexpected
                    derived_classes = [f"class_{i}" for i in range(len(reference_torchgeo_dataset.classes))]
            elif isinstance(reference_torchgeo_dataset.classes, list): # General case for list of strings or similar
                 derived_classes = reference_torchgeo_dataset.classes

        if derived_classes:
            print(f"classes before were {self.classes}, now derived from torchgeo dataset: {derived_classes}")
            self.classes = derived_classes
        elif classes: # Use from YAML if derivation failed but YAML provided
            print(f"classes before were {self.classes}, now using from YAML: {classes}")
            self.classes = classes
        # else self.classes remains as initially set by super() (potentially empty list)

        # Update self.num_classes (Number of classes)
        if self.classes: # If self.classes is now populated
            self.num_classes = len(self.classes)
        elif num_classes is not None: # Fallback to YAML num_classes if classes list is still empty
            self.num_classes = num_classes
        # else self.num_classes remains as initially set by super() (potentially 0)
        
        # Update self.ignore_index
        if ignore_index is not None:
            self.ignore_index = ignore_index
            
        # Update self.bands (Dictionary of modalities and band names)
        derived_bands = None
        if hasattr(reference_torchgeo_dataset, 'bands') and reference_torchgeo_dataset.bands:
            derived_bands = {}
            for band in reference_torchgeo_dataset.bands:
                modality_found = False
                for modality, patterns in band_to_modality.items():
                    for pattern in patterns:
                        if pattern.match(band):
                            if modality not in derived_bands:
                                derived_bands[modality] = []
                            derived_bands[modality].append(band)
                            modality_found = True
                            break
                    if modality_found:
                        break
                if not modality_found:
                    print(f"Warning: Band '{band}' not matched to any modality. Defaulting to 'optical'.")
                    if "optical" not in derived_bands:
                        derived_bands["optical"] = []
                    derived_bands["optical"].append(band)
            logging.getLogger().info(f"Derived bands from torchgeo dataset: {derived_bands}")
        
        if derived_bands:
            self.bands = derived_bands
        elif bands: # Use from YAML if derivation failed
            self.bands = bands
        # else self.bands remains as initially set by super()

        # Data statistics (mean, std, min, max) - these typically come from config and are modality/band specific
        if data_mean is not None: self.data_mean = data_mean
        if data_std is not None: self.data_std = data_std
        if data_min is not None: self.data_min = data_min
        if data_max is not None: self.data_max = data_max

        # --- Step 3: Prepare the actual dataset instance for __len__ and __getitem__ ---
        target_split_name = self.split # The split this TorchGeoAdapter instance should represent

        if target_split_name in ["train", "val"] and self.torchgeo_val_ratio is not None and 0 < self.torchgeo_val_ratio < 1:
            base_split_for_splitting = 'train' # Assume 'train' is the superset to be split.
                                            # TODO: Make this configurable if needed (e.g. via torchgeo_dataset_kwargs)

            base_load_kwargs = _torchgeo_dataset_kwargs.copy()
            base_load_kwargs['split'] = base_split_for_splitting

            source_dataset_for_splitting = self.torchgeo_dataset_class(
                root=self.root_path,
                download=auto_download,
                **base_load_kwargs
            )
            
            num_total = len(source_dataset_for_splitting)
            num_val = int(self.torchgeo_val_ratio * num_total)
            num_train = num_total - num_val

            if num_train <= 0 or num_val <= 0:
                raise ValueError(
                    f"Train/Val split resulted in zero samples for one split. "
                    f"Total: {num_total}, Val ratio: {self.torchgeo_val_ratio}, "
                    f"Train samples: {num_train}, Val samples: {num_val}. "
                    f"Adjust torchgeo_val_ratio or check dataset size."
                )

            generator = torch.Generator().manual_seed(self.cfg_seed if self.cfg_seed is not None else 42)
            
            train_subset, val_subset = random_split(source_dataset_for_splitting, [num_train, num_val], generator=generator)

            if target_split_name == "train":
                self.torchgeo_dataset = train_subset
            else: # target_split_name == "val"
                self.torchgeo_dataset = val_subset
        else:
            # No custom splitting. Load the target_split_name directly.
            # Also handles the 'test' split.
            final_load_kwargs = _torchgeo_dataset_kwargs.copy()
            # Use the target_split_name, but allow override from torchgeo_dataset_kwargs if 'split' is explicitly set there
            final_load_kwargs['split'] = _torchgeo_dataset_kwargs.get('split', target_split_name)


            self.torchgeo_dataset = self.torchgeo_dataset_class(
                root=self.root_path,
                download=auto_download,
                **final_load_kwargs
            )

    def __len__(self) -> int:
        if not hasattr(self, 'torchgeo_dataset'):
            return 0 # Dataset not loaded yet
        return len(self.torchgeo_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        try:
            data = self.torchgeo_dataset[index] # data is a dict from torchgeo dataset
        except IndexError:
            raise IndexError(f"Index {index} out of bounds for torchgeo dataset with length {len(self.torchgeo_dataset)}")

        if not isinstance(data, dict):
            raise TypeError(f"TorchGeo dataset item expected to be a dict, got {type(data)}")

        if "image" not in data:
            raise ValueError("TorchGeo sample missing 'image' key.")
        image_tensor = data["image"]
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"Expected image to be a torch.Tensor, got {type(image_tensor)}")

        # Determine primary modality (Pangaea's structure is image: {modality_name: tensor})

        if self.bands and isinstance(self.bands, dict) and len(self.bands.keys()) > 0:
            primary_modality = list(self.bands.keys())[0]
        else:
            primary_modality = "optical" 

        # Pangaea preprocessors expect image shape (C, T, H, W)
        # T=1 for single timeframe data (self.multi_temporal is set to 1 if config is False/None).
        # TorchGeo datasets typically return (C, H, W) for a single sample.
        if image_tensor.ndim == 3:  # (C, H, W)
            # Add a time dimension T=1: (C, 1, H, W)
            image_tensor = image_tensor.unsqueeze(1)
        elif image_tensor.ndim == 2: # Grayscale image (H, W)
            # Add channel C=1 and time T=1: (1, 1, H, W)
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(1)
        elif image_tensor.ndim == 4:
            # Input is already 4D. Assume it's (C, T, H, W) or (T, C, H, W).
            # Pangaea expects (C, T, H, W).
            # self.multi_temporal is the expected number of time steps T for Pangaea.
            if image_tensor.shape[1] == self.multi_temporal: # Already (C, T, H, W)
                pass
            elif image_tensor.shape[0] == self.multi_temporal: # Input is (T, C, H, W)
                image_tensor = image_tensor.permute(1, 0, 2, 3) # Convert to (C, T, H, W)
            else:
                # This case occurs if the image is 4D but neither C,T,H,W nor T,C,H,W matches self.multi_temporal
                # For single timeframe (self.multi_temporal == 1):
                # If shape is (1,C,H,W) -> permute to (C,1,H,W)
                # If shape is (C,1,H,W) -> ok
                if self.multi_temporal == 1:
                    if image_tensor.shape[0] == 1: # (1, C, H, W)
                         image_tensor = image_tensor.permute(1,0,2,3) # (C,1,H,W)
                    elif image_tensor.shape[1] == 1: # (C,1,H,W)
                        pass # ok
                    else:
                        raise ValueError(
                            f"Image tensor from torchgeo is 4D {image_tensor.shape} and self.multi_temporal=1. "
                            f"Expected (C,1,H,W) or (1,C,H,W)."
                        )
                else: # self.multi_temporal > 1
                    raise ValueError(
                        f"Image tensor from torchgeo is 4D {image_tensor.shape}, self.multi_temporal={self.multi_temporal}. "
                        f"Expected (C, {self.multi_temporal}, H, W) or ({self.multi_temporal}, C, H, W)."
                    )
        else:
            raise ValueError(
                f"Image tensor from torchgeo has unexpected number of dimensions: {image_tensor.ndim}. Expected 2, 3, or 4."
            )

        # Target tensor processing
        # Expected output for preprocessor check_dimension: (H, W) for segmentation masks
        target_tensor = None
        if "mask" in data:
            raw_mask = data["mask"]
            if not isinstance(raw_mask, torch.Tensor):
                raw_mask = torch.as_tensor(raw_mask, dtype=torch.long)
            elif raw_mask.dtype != torch.long: # Ensure long type for class indices
                raw_mask = raw_mask.long()

            if raw_mask.ndim == 3:
                # Case 1: (C, H, W) where C is num_classes (one-hot encoding)
                # self.num_classes should be correctly set from __init__
                if self.num_classes is not None and raw_mask.shape[0] == self.num_classes and self.num_classes > 1:
                    target_tensor = torch.argmax(raw_mask, dim=0)  # Converts to (H, W)
                # Case 2: (1, H, W) where the first dim is a singleton channel (class map with channel dim)
                elif raw_mask.shape[0] == 1:
                    target_tensor = raw_mask.squeeze(0)  # Converts to (H, W)
                # Case 3: Potentially (T, H, W) if multi_temporal, or other unexpected 3D format.
                # Given multi_temporal: False in the current YAML, T should effectively be 1.
                else:
                    print(f"Warning: Mask tensor has 3 dimensions with shape {raw_mask.shape}. "
                          f"First dimension ({raw_mask.shape[0]}) does not match num_classes ({self.num_classes or 'N/A'}) or 1. "
                          "Attempting to use the first slice as (H,W). This might be incorrect if it's not a time dimension.")
                    target_tensor = raw_mask[0] # Fallback: take the first slice.
            elif raw_mask.ndim == 2:  # Already (H, W)
                target_tensor = raw_mask
            else:
                raise ValueError(
                    f"Mask tensor has an unsupported number of dimensions: {raw_mask.ndim}. Expected 2 or 3 for segmentation."
                )
        elif "label" in data: # Classification label
            label_data = data["label"]
            if not isinstance(label_data, torch.Tensor):
                target_tensor = torch.as_tensor(label_data, dtype=torch.long)
            else:
                target_tensor = label_data.long()
            
            if target_tensor.ndim == 0: # scalar label
                target_tensor = target_tensor.unsqueeze(0) # Make it [1]
            # If multi-temporal classification, target_tensor might be [T]
            # The preprocessor check_dimension is specific to "target" (mask), not "label".
        else:
            # Create a dummy target if no 'mask' or 'label' found.
            # For segmentation, it should be (H,W). For classification, (1).
            # This needs to be consistent with what the rest of the pipeline expects for a dummy.
            print(f"Warning: No 'mask' or 'label' key found in torchgeo sample for {self.dataset_name}. "
                  f"Using dummy target of shape ({self.img_size},{self.img_size}) for segmentation tasks, or [0] for others.")
            # Heuristic: if num_classes suggests segmentation (e.g. > 1 and not a huge number like ImageNet)
            if self.num_classes is not None and self.num_classes > 1 and self.num_classes < 1000 : # Arbitrary threshold
                 target_tensor = torch.zeros((self.img_size, self.img_size), dtype=torch.long)
            else: # Assume classification or unknown
                 target_tensor = torch.zeros(1, dtype=torch.long)


        metadata = {"index": index, "split": self.split}
        for key in ["bbox", "crs", "res", "path"]: # Common metadata keys from torchgeo
            if key in data:
                metadata[key] = data[key]
        
        output = {
            "image": {primary_modality: image_tensor},
            "target": target_tensor,
            "metadata": metadata,
        }
        return output

    def download(self, silent: bool = False) -> None:
        # TorchGeo handles its own downloads if auto_download=True was passed to its constructor.
        # RawGeoFMDataset's download_url/auto_download is separate.
        if not silent:
            print(
                f"TorchGeoAdapter relies on the underlying torchgeo dataset ({self.torchgeo_dataset_class.__name__}) "
                f"for downloading. Ensure 'download=True' was used during its instantiation "
                f"or that data is manually placed in: {self.root_path}"
            )
        # If you need to explicitly trigger a download method of the torchgeo dataset object:
        # if hasattr(self.torchgeo_dataset, 'download') and callable(self.torchgeo_dataset.download):
        #     try:
        #         if not silent: print(f"Attempting to call .download() on {self.torchgeo_dataset.__class__.__name__}")
        #         self.torchgeo_dataset.download()
        #     except Exception as e:
        #         if not silent: print(f"Failed to call .download() on torchgeo dataset: {e}")
        pass # Usually not needed as torchgeo downloads on __init__ or first access if configured.

    def __str__(self) -> str:
        return (f"TorchGeoAdapter(dataset_name={self.dataset_name}, "
                f"torchgeo_dataset={self.torchgeo_dataset.__class__.__name__}, "
                f"split={self.split}, root={self.root_path}, len={len(self)})")

