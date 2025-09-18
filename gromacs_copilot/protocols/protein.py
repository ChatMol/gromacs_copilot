"""
Protein simulation protocol for GROMACS Copilot
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List

from gromacs_copilot.protocols.base import BaseProtocol
from gromacs_copilot.core.enums import SimulationStage
from gromacs_copilot.config import FORCE_FIELDS


class ProteinProtocol(BaseProtocol):
    """Protocol for protein-only simulations"""
    
    def __init__(self, workspace: str = "./md_workspace", gmx_bin: str = "gmx"):
        """
        Initialize the protein simulation protocol
        
        Args:
            workspace: Directory to use as the working directory
        """
        super().__init__(workspace)
        
        # Initialize protein-specific attributes
        self.protein_file = None
        self.topology_file = None
        self.box_file = None
        self.solvated_file = None
        self.minimized_file = None
        self.equilibrated_file = None
        self.production_file = None
        self.gmx_bin = gmx_bin
        
        logging.info(f"Protein protocol initialized with workspace: {self.workspace}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the protocol
        
        Returns:
            Dictionary with protocol state information
        """
        try:
            files = os.listdir(self.workspace)
            
            # Get file sizes and modification times
            file_info = []
            for file in files:
                file_path = os.path.join(self.workspace, file)
                if os.path.isfile(file_path):
                    stats = os.stat(file_path)
                    file_info.append({
                        "name": file,
                        "size_bytes": stats.st_size,
                        "modified": time.ctime(stats.st_mtime),
                        "is_directory": False
                    })
                elif os.path.isdir(file_path):
                    file_info.append({
                        "name": file,
                        "is_directory": True,
                        "modified": time.ctime(os.path.getmtime(file_path))
                    })
            
            return {
                "success": True,
                "workspace_path": self.workspace,
                "current_stage": self.stage.name,
                "files": file_info,
                "protein_file": self.protein_file,
                "topology_file": self.topology_file,
                "box_file": self.box_file,
                "solvated_file": self.solvated_file,
                "minimized_file": self.minimized_file,
                "equilibrated_file": self.equilibrated_file,
                "production_file": self.production_file
            }
        except Exception as e:
            logging.error(f"Error getting protocol state: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "workspace_path": self.workspace,
                "current_stage": self.stage.name
            }
    
    def check_prerequisites(self) -> Dict[str, Any]:
        """
        Check if GROMACS is installed and available
        
        Returns:
            Dictionary with prerequisite check information
        """
        result = self.run_shell_command(f"{self.gmx_bin} --version", capture_output=True)
        
        if result["success"]:
            version_info = result["stdout"].strip()
            return {
                "success": True,
                "installed": True,
                "version": version_info
            }
        else:
            return {
                "success": False,
                "installed": False,
                "error": "GROMACS is not installed or not in PATH"
            }
    
    def set_protein_file(self, file_path: str) -> Dict[str, Any]:
        """
        Set and prepare the protein file for simulation
        
        Args:
            file_path: Path to the protein structure file (PDB or GRO)
            
        Returns:
            Dictionary with result information
        """
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"Protein file not found: {file_path}"
            }
        
        # Copy the protein file to the workspace if it's not already there
        basename = os.path.basename(file_path)
        self.protein_file = basename
        
        if os.path.abspath(file_path) != os.path.join(self.workspace, basename):
            copy_result = self.run_shell_command(f"cp {file_path} {self.workspace}/")
            if not copy_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to copy protein file to workspace: {copy_result['stderr']}"
                }
        
        # Create directories for topologies
        mkdir_result = self.run_shell_command("mkdir -p topologies")
        
        return {
            "success": True,
            "protein_file": self.protein_file,
            "file_path": os.path.join(self.workspace, self.protein_file)
        }
    
    def generate_topology(self, force_field: str, water_model: str = "spc") -> Dict[str, Any]:
        """
        Generate topology for the protein
        
        Args:
            force_field: Name of the force field to use
            water_model: Water model to use
            
        Returns:
            Dictionary with result information
        """
        if not self.protein_file:
            return {
                "success": False,
                "error": "No protein file has been set"
            }
        
        # Map user-friendly force field names to GROMACS internal names
        if force_field not in FORCE_FIELDS:
            return {
                "success": False,
                "error": f"Unknown force field: {force_field}. Available options: {list(FORCE_FIELDS.keys())}"
            }
        
        ff_name = FORCE_FIELDS[force_field]
        
        # Generate topology
        cmd = f"{self.gmx_bin} pdb2gmx -f {self.protein_file} -o protein.gro -p topology.top -i posre.itp -ff {ff_name} -water {water_model}"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to generate topology: {result['stderr']}"
            }
        
        self.topology_file = "topology.top"
        self.box_file = "protein.gro"
        
        return {
            "success": True,
            "topology_file": self.topology_file,
            "box_file": self.box_file,
            "force_field": force_field,
            "water_model": water_model
        }
    
    def define_simulation_box(self, distance: float = 1.0, box_type: str = "cubic") -> Dict[str, Any]:
        """
        Define the simulation box
        
        Args:
            distance: Minimum distance between protein and box edge (nm)
            box_type: Type of box (cubic, dodecahedron, octahedron)
            
        Returns:
            Dictionary with result information
        """
        if not self.box_file:
            return {
                "success": False,
                "error": "No protein structure file has been processed"
            }
        
        cmd = f"{self.gmx_bin} editconf -f {self.box_file} -o box.gro -c -d {distance} -bt {box_type}"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to define simulation box: {result['stderr']}"
            }
        
        self.box_file = "box.gro"
        
        return {
            "success": True,
            "box_file": self.box_file,
            "distance": distance,
            "box_type": box_type
        }
    
    def solvate_system(self) -> Dict[str, Any]:
        """
        Solvate the protein in water
        
        Returns:
            Dictionary with result information
        """
        if not self.box_file or not self.topology_file:
            return {
                "success": False,
                "error": "Box file or topology file not defined"
            }
        
        cmd = f"{self.gmx_bin} solvate -cp {self.box_file} -cs spc216.gro -o solvated.gro -p {self.topology_file}"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to solvate the protein: {result['stderr']}"
            }
        
        self.solvated_file = "solvated.gro"
        
        return {
            "success": True,
            "solvated_file": self.solvated_file
        }
    
    def add_ions(self, concentration: float = .15, neutral: bool = True) -> Dict[str, Any]:
        """
        Add ions to the solvated system
        
        Args:
            concentration: Salt concentration in M
            neutral: Whether to neutralize the system
            
        Returns:
            Dictionary with result information
        """
        if not self.solvated_file or not self.topology_file:
            return {
                "success": False,
                "error": "Solvated file or topology file not defined"
            }
        
        # Create ions.mdp file
        ions_mdp = self.create_mdp_file("ions")
        if not ions_mdp["success"]:
            return ions_mdp
        
        # Prepare for adding ions
        cmd = f"{self.gmx_bin} grompp -f ions.mdp -c {self.solvated_file} -p {self.topology_file} -o ions.tpr"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to prepare for adding ions: {result['stderr']}"
            }
        
        # Add ions
        neutral_flag = "-neutral" if neutral else ""
        cmd = f"echo 'SOL' | {self.gmx_bin} genion -s ions.tpr -o solvated_ions.gro -p {self.topology_file} -pname NA -nname CL {neutral_flag} -conc {concentration}"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to add ions: {result['stderr']}"
            }
        
        self.solvated_file = "solvated_ions.gro"
        
        return {
            "success": True,
            "solvated_file": self.solvated_file,
            "concentration": concentration,
            "neutral": neutral
        }
    
    def run_energy_minimization(self) -> Dict[str, Any]:
        """
        Run energy minimization
        
        Returns:
            Dictionary with result information
        """
        if not self.solvated_file or not self.topology_file:
            return {
                "success": False,
                "error": "Solvated file or topology file not defined"
            }
        
        # Create em.mdp file
        em_mdp = self.create_mdp_file("em")
        if not em_mdp["success"]:
            return em_mdp
        
        # Generate tpr file for minimization
        cmd = f"{self.gmx_bin} grompp -f em.mdp -c {self.solvated_file} -p {self.topology_file} -o em.tpr"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to prepare energy minimization: {result['stderr']}"
            }
        
        # Run energy minimization
        cmd = f"{self.gmx_bin} mdrun -v -deffnm em"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            # return {
            #     "success": False,
            #     "error": f"Energy minimization failed: {result['stderr']}"
            # }
            cmd = f"{self.gmx_bin} mdrun -ntmpi 1 -v -deffnm em"
            result = self.run_shell_command(cmd)
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Energy minimization failed: {result['stderr']}"
                }
        
        self.minimized_file = "em.gro"
        
        return {
            "success": True,
            "minimized_file": self.minimized_file,
            "log_file": "em.log",
            "energy_file": "em.edr"
        }
    
    def run_nvt_equilibration(self) -> Dict[str, Any]:
        """
        Run NVT equilibration
        
        Returns:
            Dictionary with result information
        """
        if not self.minimized_file or not self.topology_file:
            return {
                "success": False,
                "error": "Minimized file or topology file not defined"
            }
        
        # Create nvt.mdp file
        nvt_mdp = self.create_mdp_file("nvt")
        if not nvt_mdp["success"]:
            return nvt_mdp
        
        # Generate tpr file for NVT equilibration
        cmd = f"{self.gmx_bin} grompp -f nvt.mdp -c {self.minimized_file} -r {self.minimized_file} -p {self.topology_file} -o nvt.tpr"
        # print(f"Running command: {cmd}")
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to prepare NVT equilibration: {result['stderr']}"
            }
        
        # Run NVT equilibration
        cmd = f"{self.gmx_bin} mdrun -v -deffnm nvt"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            # return {
            #     "success": False,
            #     "error": f"NVT equilibration failed: {result['stderr']}"
            # }
            cmd = f"{self.gmx_bin} mdrun -ntmpi 1 -v -deffnm nvt"
            result = self.run_shell_command(cmd)
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"NVT equilibration failed: {result['stderr']}"
                }
        
        return {
            "success": True,
            "nvt_file": "nvt.gro",
            "nvt_checkpoint": "nvt.cpt",
            "log_file": "nvt.log",
            "energy_file": "nvt.edr"
        }
    
    def run_npt_equilibration(self) -> Dict[str, Any]:
        """
        Run NPT equilibration
        
        Returns:
            Dictionary with result information
        """
        # Create npt.mdp file
        npt_mdp = self.create_mdp_file("npt")
        if not npt_mdp["success"]:
            return npt_mdp
        
        # Generate tpr file for NPT equilibration
        cmd = f"{self.gmx_bin} grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p {self.topology_file} -o npt.tpr"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to prepare NPT equilibration: {result['stderr']}"
            }
        
        # Run NPT equilibration
        cmd = f"{self.gmx_bin} mdrun -v -deffnm npt"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            # return {
            #     "success": False,
            #     "error": f"NPT equilibration failed: {result['stderr']}"
            # }
            cmd = f"{self.gmx_bin} mdrun -ntmpi 1 -v -deffnm npt"
            result = self.run_shell_command(cmd)
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"NPT equilibration failed: {result['stderr']}"
                }
        
        self.equilibrated_file = "npt.gro"
        
        return {
            "success": True,
            "equilibrated_file": self.equilibrated_file,
            "npt_checkpoint": "npt.cpt",
            "log_file": "npt.log",
            "energy_file": "npt.edr"
        }
    
    def run_production_md(self, length_ns: float = 10.0) -> Dict[str, Any]:
        """
        Run production MD
        
        Args:
            length_ns: Length of the simulation in nanoseconds
            
        Returns:
            Dictionary with result information
        """
        if not self.equilibrated_file or not self.topology_file:
            return {
                "success": False,
                "error": "Equilibrated file or topology file not defined"
            }
        
        # Calculate number of steps (2 fs timestep)
        nsteps = int(length_ns * 1000000 / 2)
        
        # Create md.mdp file with custom steps
        md_mdp = self.create_mdp_file("md", {"nsteps": nsteps})
        if not md_mdp["success"]:
            return md_mdp
        
        # Generate tpr file for production MD
        cmd = f"{self.gmx_bin} grompp -f md.mdp -c {self.equilibrated_file} -t npt.cpt -p {self.topology_file} -o md.tpr"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to prepare production MD: {result['stderr']}"
            }
        
        # Run production MD
        cmd = f"{self.gmx_bin} mdrun -v -deffnm md"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            # return {
            #     "success": False,
            #     "error": f"Production MD failed: {result['stderr']}"
            # }
            cmd = f"{self.gmx_bin} mdrun -ntmpi 1 -v -deffnm md"
            result = self.run_shell_command(cmd)
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Production MD failed: {result['stderr']}"
                }
        
        self.production_file = "md.gro"
        
        return {
            "success": True,
            "production_file": self.production_file,
            "trajectory_file": "md.xtc",
            "log_file": "md.log",
            "energy_file": "md.edr",
            "length_ns": length_ns
        }
    
    def analyze_rmsd(self) -> Dict[str, Any]:
        """
        Perform RMSD analysis
        
        Returns:
            Dictionary with result information
        """
        # Create analysis directory if it doesn't exist
        mkdir_result = self.run_shell_command("mkdir -p analysis")
        
        cmd = f"echo 'Protein Protein' | {self.gmx_bin} rms -s md.tpr -f md.xtc -o analysis/rmsd.xvg -tu ns"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"RMSD analysis failed: {result['stderr']}"
            }
        
        return {
            "success": True,
            "output_file": "analysis/rmsd.xvg",
            "analysis_type": "RMSD"
        }
    
    def analyze_rmsf(self) -> Dict[str, Any]:
        """
        Perform RMSF analysis
        
        Returns:
            Dictionary with result information
        """
        # Create analysis directory if it doesn't exist
        mkdir_result = self.run_shell_command("mkdir -p analysis")
        
        cmd = f"echo 'C-alpha' | {self.gmx_bin} rmsf -s md.tpr -f md.xtc -o analysis/rmsf.xvg -res"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"RMSF analysis failed: {result['stderr']}"
            }
        
        return {
            "success": True,
            "output_file": "analysis/rmsf.xvg",
            "analysis_type": "RMSF"
        }
    
    def analyze_gyration(self) -> Dict[str, Any]:
        """
        Perform radius of gyration analysis
        
        Returns:
            Dictionary with result information
        """
        # Create analysis directory if it doesn't exist
        mkdir_result = self.run_shell_command("mkdir -p analysis")
        
        cmd = f"echo 'Protein' | {self.gmx_bin} gyrate -s md.tpr -f md.xtc -o analysis/gyrate.xvg"
        result = self.run_shell_command(cmd)
        
        if not result["success"]:
            return {
                "success": False,
                "error": f"Radius of gyration analysis failed: {result['stderr']}"
            }
        
        return {
            "success": True,
            "output_file": "analysis/gyrate.xvg",
            "analysis_type": "Radius of Gyration"
        }
