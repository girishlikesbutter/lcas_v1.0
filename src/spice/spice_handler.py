import spiceypy
import numpy as np
from typing import Union, List, Tuple, Set
import logging
import os


class SpiceHandler:
    """
    Simplified SPICE handler using native furnsh capabilities.
    
    This version leverages SPICE's native metakernel handling instead of
    manually parsing files. It requires metakernels to use project-root
    relative paths or absolute paths.
    """
    
    def __init__(self):
        self._loaded_kernels: Set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("SpiceHandler instance initialized.")

    def load_metakernel(self, metakernel_path: str, project_root: str = None):
        """
        Load a SPICE metakernel using native furnsh.
        
        For new metakernels (v2), use with project_root parameter.
        For legacy metakernels, use load_metakernel_programmatically.
        
        Args:
            metakernel_path: Path to the metakernel file
            project_root: Optional project root for resolving paths. 
                         If not provided, uses current working directory.
        """
        if metakernel_path in self._loaded_kernels:
            self.logger.debug(f"Metakernel already loaded: {metakernel_path}")
            return
            
        # Save current directory
        original_cwd = os.getcwd()
        
        try:
            # Change to project root if specified
            if project_root:
                os.chdir(project_root)
                self.logger.debug(f"Changed directory to: {project_root}")
            
            # Use SPICE's native furnsh - it handles everything automatically
            spiceypy.furnsh(metakernel_path)
            self._loaded_kernels.add(metakernel_path)
            self.logger.info(f"Loaded SPICE metakernel: {metakernel_path}")
            
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error loading metakernel {metakernel_path}: {e}")
            raise
        finally:
            # Always restore original directory
            os.chdir(original_cwd)

    def load_metakernel_programmatically(self, metakernel_path: str):
        """
        Legacy method for backward compatibility with old metakernels.
        
        This method attempts to determine the project root from the metakernel
        path and loads it with proper path resolution.
        """
        # Try to detect if this is a v2 metakernel
        if '-v2.tm' in metakernel_path:
            # Extract project root from metakernel path
            parts = os.path.normpath(metakernel_path).split(os.sep)
            if 'data' in parts:
                idx = parts.index('data')
                project_root = os.sep.join(parts[:idx]) or '.'
                self.load_metakernel(metakernel_path, project_root)
                return
        
        # For legacy metakernels, we need special handling
        # This is a simplified version that changes to metakernel directory
        self.logger.warning(
            "Using legacy metakernel format. Consider updating to v2 format "
            "with project-root relative paths for better portability."
        )
        
        metakernel_dir = os.path.dirname(os.path.abspath(metakernel_path))
        original_cwd = os.getcwd()
        
        try:
            # Change to metakernel directory for legacy relative paths
            os.chdir(metakernel_dir)
            self.logger.debug(f"Changed to metakernel directory: {metakernel_dir}")
            
            # Load using relative path from metakernel directory
            mk_filename = os.path.basename(metakernel_path)
            spiceypy.furnsh(mk_filename)
            self._loaded_kernels.add(metakernel_path)
            self.logger.info(f"Loaded legacy metakernel: {metakernel_path}")
            
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error loading legacy metakernel {metakernel_path}: {e}")
            raise
        finally:
            os.chdir(original_cwd)

    def load_kernel(self, kernel_path: Union[str, List[str]]):
        """Load individual kernel(s)."""
        if isinstance(kernel_path, str):
            kernel_paths = [kernel_path]
        else:
            kernel_paths = kernel_path
            
        for path in kernel_paths:
            if path not in self._loaded_kernels:
                try:
                    spiceypy.furnsh(path)
                    self._loaded_kernels.add(path)
                    self.logger.info(f"Loaded SPICE kernel: {path}")
                except spiceypy.utils.exceptions.SpiceyError as e:
                    self.logger.error(f"SPICE error loading kernel {path}: {e}")
                    raise
            else:
                self.logger.debug(f"Kernel already loaded, skipping: {path}")

    def unload_kernel(self, kernel_path: str):
        """Unload a specific kernel."""
        if kernel_path in self._loaded_kernels:
            try:
                spiceypy.unload(kernel_path)
                self._loaded_kernels.remove(kernel_path)
                self.logger.info(f"Unloaded SPICE kernel: {kernel_path}")
            except spiceypy.utils.exceptions.SpiceyError as e:
                self.logger.error(f"SPICE error unloading kernel {kernel_path}: {e}")
                raise
        else:
            self.logger.warning(f"Attempted to unload kernel not in loaded set: {kernel_path}")

    def unload_all_kernels(self):
        """Clear all loaded kernels."""
        try:
            spiceypy.kclear()
            self._loaded_kernels.clear()
            self.logger.info("All SPICE kernels unloaded and pool cleared (kclear).")
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error during kclear: {e}")
            raise

    def utc_to_et(self, utc_time_str: str) -> float:
        """Convert UTC time string to ephemeris time."""
        try:
            et = spiceypy.utc2et(utc_time_str)
            self.logger.debug(f"Converted UTC '{utc_time_str}' to ET {et}.")
            return et
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error converting UTC '{utc_time_str}' to ET: {e}")
            raise

    def et_to_utc(self, et: float, time_format: str = "ISOC", precision: int = 3) -> str:
        """Convert ephemeris time to UTC string."""
        try:
            utc_str = spiceypy.et2utc(et, time_format, precision)
            self.logger.debug(f"Converted ET {et} to UTC '{utc_str}'.")
            return utc_str
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error converting ET {et} to UTC: {e}")
            raise

    def get_body_position(self, target: str, et: float, frame: str, observer: str,
                          aberration_correction: str = 'NONE') -> Tuple[np.ndarray, float]:
        """Get position of target body relative to observer."""
        try:
            position_vector, light_time = spiceypy.spkpos(
                targ=str(target), et=et, ref=frame,
                abcorr=aberration_correction, obs=str(observer)
            )
            self.logger.debug(
                f"Position of '{target}' relative to '{observer}' in '{frame}' "
                f"at ET {et}: {position_vector}, LT: {light_time}s."
            )
            return np.array(position_vector), light_time
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(
                f"SPICE error getting position for target='{target}', "
                f"observer='{observer}', frame='{frame}', et={et}: {e}"
            )
            raise

    def get_target_orientation(self, from_frame: str, to_frame: str, et: float) -> np.ndarray:
        """Get rotation matrix from one frame to another."""
        try:
            self.logger.debug(f"Attempting pxform from '{from_frame}' to '{to_frame}' at ET {et}")
            rotation_matrix = spiceypy.pxform(from_frame, to_frame, et)
            self.logger.info(f"Successfully obtained rotation matrix from '{from_frame}' to '{to_frame}'.")
            return np.array(rotation_matrix)
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(
                f"SPICE error getting orientation matrix for from_frame='{from_frame}', "
                f"to_frame='{to_frame}', et={et}: {e}"
            )
            raise

    def get_frame_name_from_id(self, frame_id: int) -> str:
        """Get frame name from frame ID."""
        try:
            frame_name = spiceypy.frmnam(frame_id)
            if not frame_name:
                raise ValueError(f"No frame name found for ID {frame_id}")
            self.logger.debug(f"Frame ID {frame_id} resolved to name '{frame_name}'.")
            return frame_name
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error getting frame name for ID {frame_id}: {e}")
            raise ValueError(f"Could not get frame name for ID {frame_id}: {e}")

    def get_frame_id_from_name(self, frame_name: str) -> int:
        """Get frame ID from frame name."""
        try:
            frame_id = spiceypy.namfrm(frame_name)
            if frame_id == 0:
                raise ValueError(f"No frame ID found for name '{frame_name}'.")
            self.logger.debug(f"Frame name '{frame_name}' resolved to ID {frame_id}.")
            return frame_id
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error getting frame ID for name '{frame_name}': {e}")
            raise ValueError(f"Could not get frame ID for name '{frame_name}': {e}")

    def get_frame_info_by_id(self, frame_id: int) -> Tuple[str, int, int, List[int]]:
        """Get frame information by frame ID."""
        try:
            # Get frame name
            frame_name = spiceypy.frmnam(frame_id)
            if not frame_name:
                self.logger.warning(f"No frame name found for ID {frame_id}")
                frame_name = ""
            
            # Get frame info
            center, frclass, clssid = spiceypy.frinfo(frame_id)
            
            # Convert class ID to list format for compatibility
            frclss_id_list = [clssid] if clssid != 0 else []
            
            self.logger.debug(
                f"Frame Info for ID {frame_id}: Name='{frame_name}', "
                f"Center={center}, Class={frclass}, ClassIDList={frclss_id_list}"
            )
            
            return frame_name, center, frclass, frclss_id_list
            
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error getting frame info for ID {frame_id}: {e}")
            raise ValueError(f"Error getting frame info for ID {frame_id}: {e}")

    def __enter__(self):
        self.logger.debug("SpiceHandler context entered.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all_kernels()
        self.logger.debug("SpiceHandler context exited, all kernels unloaded.")


# Example usage
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    main_logger = logging.getLogger(f"{__name__}.__main__")
    main_logger.info("Running SpiceHandler example.")
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Test with both old and new metakernels
    old_metakernel = os.path.join(
        project_root, "data", "spice_kernels", "missions", 
        "dst-is901", "INTELSAT_901-metakernel.tm"
    )
    new_metakernel = os.path.join(
        project_root, "data", "spice_kernels", "missions", 
        "dst-is901", "INTELSAT_901-metakernel-v2.tm"
    )
    
    # Test data
    TEST_ET = 6.34169191e+08
    IS901_ID = "-126824"
    EARTH_ID = "399"
    J2000_FRAME = "J2000"
    IS901_BUS_FRAME = "IS901_BUS_FRAME"
    
    main_logger.info("Testing with new v2 metakernel (recommended)...")
    with SpiceHandler() as sh:
        sh.load_metakernel(new_metakernel, project_root)
        
        utc = sh.et_to_utc(TEST_ET, precision=6)
        main_logger.info(f"ET {TEST_ET} -> UTC: {utc}")
        
        pos, lt = sh.get_body_position(IS901_ID, TEST_ET, J2000_FRAME, EARTH_ID, "LT+S")
        main_logger.info(f"IS901 position: {pos} km (LT: {lt:.6f}s)")
    
    main_logger.info("\nTesting with legacy metakernel (backward compatibility)...")
    with SpiceHandler() as sh:
        sh.load_metakernel_programmatically(old_metakernel)
        
        utc = sh.et_to_utc(TEST_ET, precision=6)
        main_logger.info(f"ET {TEST_ET} -> UTC: {utc}")
        
        pos, lt = sh.get_body_position(IS901_ID, TEST_ET, J2000_FRAME, EARTH_ID, "LT+S")
        main_logger.info(f"IS901 position: {pos} km (LT: {lt:.6f}s)")
    
    main_logger.info("SpiceHandler example finished.")