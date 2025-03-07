"""
Base protocol class for GROMACS Copilot
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from gromacs_copilot.utils.shell import run_shell_command
from gromacs_copilot.core.enums import SimulationStage


class BaseProtocol(ABC):
    """Base class for simulation protocols"""
    
    def __init__(self, workspace: str = "./md_workspace"):
        """
        Initialize the base protocol
        
        Args:
            workspace: Directory to use as the working directory
        """
        self.workspace = os.path.abspath(workspace)
        self.stage = SimulationStage.SETUP
        
        # Create workspace if it doesn't exist
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        
        # Change to workspace directory
        os.chdir(self.workspace)
        
        logging.info(f"Protocol initialized with workspace: {self.workspace}")
    
    def run_shell_command(self, command: str, capture_output: bool = True,
                         suppress_output: bool = False) -> Dict[str, Any]:
        """
        Run a shell command
        
        Args:
            command: Shell command to run
            capture_output: Whether to capture stdout/stderr
            suppress_output: Whether to suppress terminal output
            
        Returns:
            Dictionary with command result information
        """
        return run_shell_command(command, capture_output, suppress_output)
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the protocol
        
        Returns:
            Dictionary with protocol state information
        """
        pass
    
    @abstractmethod
    def check_prerequisites(self) -> Dict[str, Any]:
        """
        Check if prerequisites for the protocol are met
        
        Returns:
            Dictionary with prerequisite check information
        """
        pass
    
    def create_mdp_file(self, mdp_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an MDP parameter file for GROMACS
        
        Args:
            mdp_type: Type of MDP file (ions, em, nvt, npt, md)
            params: Optional override parameters
            
        Returns:
            Dictionary with result information
        """
        from gromacs_copilot.config import DEFAULT_MDP_PARAMS, MDP_TYPES
        
        if mdp_type not in MDP_TYPES:
            return {
                "success": False,
                "error": f"Unknown MDP type: {mdp_type}. Available types: {MDP_TYPES}"
            }
        
        # Start with default parameters for the specified type
        mdp_params = DEFAULT_MDP_PARAMS[mdp_type].copy()
        
        # Override with user-provided parameters if any
        if params:
            mdp_params.update(params)
        
        # Create MDP file content
        mdp_content = f"; {mdp_type}.mdp - Generated by GROMACS Copilot\n"
        for key, value in mdp_params.items():
            mdp_content += f"{key:<20} = {value}\n"
        
        # Write MDP file
        file_path = f"{mdp_type}.mdp"
        try:
            with open(file_path, "w") as f:
                f.write(mdp_content)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write MDP file: {str(e)}"
            }
        
        return {
            "success": True,
            "file_path": file_path,
            "mdp_type": mdp_type,
            "params": mdp_params
        }
    
    def set_simulation_stage(self, stage: str) -> Dict[str, Any]:
        """
        Set the current simulation stage
        
        Args:
            stage: Name of the stage to set
            
        Returns:
            Dictionary with result information
        """
        try:
            self.stage = SimulationStage[stage]
            return {
                "success": True,
                "stage": self.stage.name,
                "previous_stage": self.stage.name
            }
        except KeyError:
            return {
                "success": False,
                "error": f"Unknown stage: {stage}. Available stages: {[s.name for s in SimulationStage]}"
            }