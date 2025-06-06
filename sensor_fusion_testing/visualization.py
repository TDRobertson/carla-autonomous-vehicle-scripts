import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import seaborn as sns
import os
from dataclasses import dataclass
from gps_spoofer import SpoofingStrategy

@dataclass
class PlotConfig:
    figsize: tuple = (10, 6)
    dpi: int = 100
    max_figures: int = 10  # Maximum number of figures to keep open

class DataVisualizer:
    def __init__(self):
        self.figures = {}
        self.config = PlotConfig()
        plt.rcParams['figure.max_open_warning'] = self.config.max_figures
        
    def _create_figure(self, name: str) -> plt.Figure:
        """Create a new figure, closing old ones if needed."""
        # Close the figure if it already exists
        if name in self.figures:
            plt.close(self.figures[name])
            
        # If we have too many figures open, close the oldest one
        if len(self.figures) >= self.config.max_figures:
            oldest_figure = next(iter(self.figures.values()))
            plt.close(oldest_figure)
            # Remove the oldest figure from our dictionary
            for key in list(self.figures.keys()):
                if self.figures[key] == oldest_figure:
                    del self.figures[key]
                    break
                    
        # Create new figure
        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        self.figures[name] = fig
        return fig
        
    def plot_position_tracking(self, true_positions: List[np.ndarray], 
                             fused_positions: List[np.ndarray],
                             timestamps: List[float]) -> Optional[plt.Figure]:
        """Plot true vs fused position tracking."""
        if not true_positions or not fused_positions:
            return None
            
        fig = self._create_figure('position_tracking')
        ax = fig.add_subplot(111)
        
        true_pos = np.array(true_positions)
        fused_pos = np.array(fused_positions)
        
        ax.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True Position')
        ax.plot(fused_pos[:, 0], fused_pos[:, 1], 'r--', label='Fused Position')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Position Tracking')
        ax.legend()
        ax.grid(True)
        
        return fig
        
    def plot_error_evolution(self, position_errors: List[float], 
                           velocity_errors: List[float],
                           timestamps: List[float]) -> Optional[plt.Figure]:
        """Plot position and velocity error evolution."""
        if not position_errors or not velocity_errors:
            return None
            
        fig = self._create_figure('error_evolution')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        ax1.plot(timestamps, position_errors, 'b-', label='Position Error')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error (m)')
        ax1.set_title('Position Error Evolution')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(timestamps, velocity_errors, 'r-', label='Velocity Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m/s)')
        ax2.set_title('Velocity Error Evolution')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        return fig
        
    def plot_velocity_profiles(self, true_velocities: List[np.ndarray],
                             fused_velocities: List[np.ndarray],
                             timestamps: List[float]) -> Optional[plt.Figure]:
        """Plot true vs fused velocity profiles."""
        if not true_velocities or not fused_velocities:
            return None
            
        # Ensure all arrays have the same length
        min_len = min(len(true_velocities), len(fused_velocities), len(timestamps))
        if min_len == 0:
            return None
            
        true_vel = np.array(true_velocities[:min_len])
        fused_vel = np.array(fused_velocities[:min_len])
        times = np.array(timestamps[:min_len])
        
        fig = self._create_figure('velocity_profiles')
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Plot X velocity
        ax1.plot(times, true_vel[:, 0], 'b-', label='True X Velocity')
        ax1.plot(times, fused_vel[:, 0], 'r--', label='Fused X Velocity')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('X Velocity Profile')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Y velocity
        ax2.plot(times, true_vel[:, 1], 'b-', label='True Y Velocity')
        ax2.plot(times, fused_vel[:, 1], 'r--', label='Fused Y Velocity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Y Velocity Profile')
        ax2.legend()
        ax2.grid(True)
        
        fig.tight_layout()
        return fig
        
    def plot_error_distribution(self, errors: List[float]) -> Optional[plt.Figure]:
        """Plot error distribution histogram."""
        if not errors:
            return None
            
        fig = self._create_figure('error_distribution')
        ax = fig.add_subplot(111)
        
        ax.hist(errors, bins=50, density=True, alpha=0.7)
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Density')
        ax.set_title('Position Error Distribution')
        ax.grid(True)
        
        return fig
        
    def plot_position_error_heatmap(self, true_positions: List[np.ndarray],
                                  fused_positions: List[np.ndarray]) -> Optional[plt.Figure]:
        """Plot position error heatmap."""
        if not true_positions or not fused_positions:
            return None
            
        true_pos = np.array(true_positions)
        fused_pos = np.array(fused_positions)
        
        # Calculate error magnitude at each point
        errors = np.linalg.norm(true_pos - fused_pos, axis=1)
        
        fig = self._create_figure('error_heatmap')
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(true_pos[:, 0], true_pos[:, 1], c=errors, cmap='hot')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Position Error Heatmap')
        fig.colorbar(scatter, label='Error (m)')
        ax.grid(True)
        
        return fig
        
    def plot_attack_transitions(self, attack_types: List[SpoofingStrategy],
                              timestamps: List[float],
                              config: PlotConfig = None):
        """Plot attack type transitions."""
        if config is None:
            config = PlotConfig(
                title="Attack Type Transitions",
                xlabel="Time (s)",
                ylabel="Attack Type"
            )
            
        plt.figure(figsize=config.figsize)
        unique_attacks = list(set(attack_types))
        attack_indices = [unique_attacks.index(a) for a in attack_types]
        
        plt.plot(timestamps, attack_indices, 'b-')
        plt.yticks(range(len(unique_attacks)), [str(a) for a in unique_attacks])
        plt.title(config.title)
        plt.xlabel(config.xlabel)
        plt.ylabel(config.ylabel)
        plt.grid(config.grid)
        return plt.gcf()
        
    def plot_correlation_matrix(self, metrics: Dict[str, List[float]]) -> Optional[plt.Figure]:
        """Plot correlation matrix of all metrics."""
        if not metrics:
            return None
            
        # Find the minimum length among all metric arrays
        min_length = min(len(values) for values in metrics.values())
        if min_length == 0:
            return None
            
        # Truncate all arrays to the minimum length
        metric_names = list(metrics.keys())
        metric_values = np.array([metrics[name][:min_length] for name in metric_names])
        
        # Calculate correlation matrix
        try:
            corr_matrix = np.corrcoef(metric_values)
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return None
            
        fig = self._create_figure('correlation_matrix')
        ax = fig.add_subplot(111)
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_names)
        
        # Add colorbar
        fig.colorbar(im, ax=ax)
        
        ax.set_title('Metric Correlation Matrix')
        fig.tight_layout()
        return fig
        
    def save_all_plots(self, output_dir: str):
        """Save all plots to the specified directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for name, fig in self.figures.items():
            if fig is not None:
                try:
                    fig.savefig(os.path.join(output_dir, f"{name}.png"))
                except Exception as e:
                    print(f"Error saving figure {name}: {e}")
                    
    def show_all_plots(self):
        """Display all generated plots."""
        plt.show()
        
    def clear_plots(self):
        """Clear all stored plots and close figures."""
        for fig in self.figures.values():
            if fig is not None:
                plt.close(fig)
        self.figures.clear()
        
    def _update_visualizations(self, current_strategy: SpoofingStrategy):
        """Update visualizations for the current attack strategy."""
        # Clear previous figures for this strategy
        for name in list(self.figures.keys()):
            if name.startswith(f'position_tracking_{current_strategy}'):
                plt.close(self.figures[name])
                del self.figures[name]
                
        results = self.results[current_strategy]
        
        # Position tracking plot
        fig = self.plot_position_tracking(
            results['true_positions'],
            results['fused_positions'],
            results['timestamps']
        )
        if fig is not None:
            self.figures[f'position_tracking_{current_strategy}'] = fig
            
        # Error evolution plot
        fig = self.plot_error_evolution(
            results['position_errors'],
            results['velocity_errors'],
            results['timestamps']
        )
        if fig is not None:
            self.figures[f'error_evolution_{current_strategy}'] = fig
            
        # Velocity profiles plot
        if len(results['true_velocities']) > 0 and len(results['fused_velocities']) > 0:
            fig = self.plot_velocity_profiles(
                results['true_velocities'],
                results['fused_velocities'],
                results['timestamps']
            )
            if fig is not None:
                self.figures[f'velocity_profiles_{current_strategy}'] = fig
                
        # Error distribution plot
        fig = self.plot_error_distribution(
            results['position_errors']
        )
        if fig is not None:
            self.figures[f'error_distribution_{current_strategy}'] = fig
            
        # Position error heatmap
        fig = self.plot_position_error_heatmap(
            results['true_positions'],
            results['fused_positions']
        )
        if fig is not None:
            self.figures[f'error_heatmap_{current_strategy}'] = fig
            
        # Correlation matrix
        fig = self.plot_correlation_matrix(
            {f'Metric {i+1}': [metric] for i, metric in enumerate(results['metrics'])}
        )
        if fig is not None:
            self.figures[f'correlation_matrix_{current_strategy}'] = fig 