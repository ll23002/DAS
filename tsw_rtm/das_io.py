"""
DAS I/O Module for Neubrescope HDF5 Strain Rate records.
Reads raw fiber acoustic/strain rate data, extracts acquisition geometry,
and provides lazy loading, channel windowing, and multi-file folder batch processing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Generator
import glob
import os
import h5py
import numpy as np


@dataclass
class DASMetadata:
    filename: str
    num_samples: int
    num_channels: int
    dt: float  # seconds
    sampling_rate: float  # Hz
    dx: float  # meters (SpatialSamplingInterval)
    gauge_length: float  # meters
    unit: str
    data_type: str
    timestamp: str
    distance_range: float  # total meters


class DASReader:
    """Reader for a single Neubrescope NBX-S4110 HDF5 DAS file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.metadata = self._read_metadata()

    def _read_metadata(self) -> DASMetadata:
        with h5py.File(self.filepath, 'r') as f:
            if 'Acoustic' not in f:
                raise KeyError(f"Dataset 'Acoustic' not found in {self.filepath}")
            ds = f['Acoustic']
            attrs = ds.attrs
            num_samples, num_channels = ds.shape

            dt = float(attrs.get('TimeSamplingInterval(seconds)', 0.001))
            fs = float(attrs.get('TimeSamplingRate(Hz)', 1.0 / dt if dt > 0 else 1000.0))
            dx = float(attrs.get('SpatialSamplingInterval', 5.0))
            gl = float(attrs.get('GaugeLength', dx))
            unit = str(attrs.get('DataPhysicalUnit', 'micro-strain/second'))
            dtype = str(attrs.get('DataType', 'strain-rate'))
            ts = str(attrs.get('ISO8601TimeStamp', ''))
            dist_range = float(attrs.get('DistanceRange', num_channels * dx))

            return DASMetadata(
                filename=self.filepath,
                num_samples=num_samples,
                num_channels=num_channels,
                dt=dt,
                sampling_rate=fs,
                dx=dx,
                gauge_length=gl,
                unit=unit,
                data_type=dtype,
                timestamp=ts,
                distance_range=dist_range,
            )

    def read_window(
        self,
        channel_min: int = 0,
        channel_max: Optional[int] = None,
        sample_min: int = 0,
        sample_max: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads a spatial and temporal window of DAS data.

        Returns:
            data: np.ndarray of shape (num_selected_samples, num_selected_channels)
            time_axis: 1D array of time in seconds
            distance_axis: 1D array of fiber distance in meters
        """
        ch_end = self.metadata.num_channels if channel_max is None else min(channel_max, self.metadata.num_channels)
        samp_end = self.metadata.num_samples if sample_max is None else min(sample_max, self.metadata.num_samples)

        if channel_min < 0 or channel_min >= ch_end:
            raise ValueError(f"Invalid channel range [{channel_min}, {ch_end})")
        if sample_min < 0 or sample_min >= samp_end:
            raise ValueError(f"Invalid sample range [{sample_min}, {samp_end})")

        with h5py.File(self.filepath, 'r') as f:
            ds = f['Acoustic']
            data = ds[sample_min:samp_end, channel_min:ch_end].astype(np.float32)

        time_axis = np.arange(sample_min, samp_end) * self.metadata.dt
        dist_axis = np.arange(channel_min, ch_end) * self.metadata.dx

        return data, time_axis, dist_axis


class DASFolderReader:
    """
    Reader for continuous multi-file DAS datasets stored in a folder.
    Sorts files chronologically and yields preprocessed chunks or windows.
    """

    def __init__(self, folder_path: str, pattern: str = "*.h5"):
        self.folder_path = os.path.abspath(folder_path)
        search_path = os.path.join(self.folder_path, pattern)
        self.file_list = sorted(glob.glob(search_path))

        if not self.file_list:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {self.folder_path}")

        # Read metadata of the first file as reference
        self.first_reader = DASReader(self.file_list[0])
        self.metadata = self.first_reader.metadata
        print(f"DASFolderReader initialized: {len(self.file_list)} files found in {self.folder_path}")
        print(f"Channels: {self.metadata.num_channels}, Fs: {self.metadata.sampling_rate} Hz, dx: {self.metadata.dx:.2f} m")

    def __len__(self) -> int:
        return len(self.file_list)

    def iterate_windows(
        self,
        channel_min: int = 0,
        channel_max: Optional[int] = None,
        max_files: Optional[int] = None
    ) -> Generator[Tuple[str, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Iterates over each file in chronological order, yielding:
        (filepath, windowed_data, time_axis, dist_axis)
        """
        files_to_read = self.file_list if max_files is None else self.file_list[:max_files]
        for idx, fpath in enumerate(files_to_read):
            reader = DASReader(fpath)
            data, t_ax, d_ax = reader.read_window(
                channel_min=channel_min,
                channel_max=channel_max
            )
            yield fpath, data, t_ax, d_ax
