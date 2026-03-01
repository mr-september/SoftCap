""
Experiment Utilities

This module provides utility functions for managing and running experiments
in the SoftCap research framework.
"""

import os
import json
import yaml
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union

from experiments.base.base_experiment import BaseExperiment


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            return json.load(f)
        elif config_path.suffix.lower() in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save experiment configuration to a JSON or YAML file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration to
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        elif config_path.suffix.lower() in ('.yaml', '.yml'):
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def get_experiment_class(experiment_path: str) -> Type[BaseExperiment]:
    """
    Dynamically import and return an experiment class.
    
    Args:
        experiment_path: Dotted path to the experiment class
                        (e.g., 'experiments.cv.mnist_experiment.MNISTExperiment')
    
    Returns:
        The experiment class
    """
    module_path, class_name = experiment_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_experiment(
    experiment_class: Union[str, Type[BaseExperiment]],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an experiment with the given configuration.
    
    Args:
        experiment_class: Experiment class or dotted path to the class
        config: Experiment configuration dictionary
        **kwargs: Additional arguments to pass to the experiment
        
    Returns:
        Dictionary containing experiment results
    """
    # Load experiment class if a string path is provided
    if isinstance(experiment_class, str):
        experiment_class = get_experiment_class(experiment_class)
    
    # Merge config with kwargs (kwargs take precedence)
    if config is None:
        config = {}
    config.update(kwargs)
    
    # Create and run experiment
    experiment = experiment_class(**config)
    results = experiment.run()
    
    return results


def run_experiment_from_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Run an experiment from a configuration file.
    
    Args:
        config_path: Path to the experiment configuration file
        
    Returns:
        Dictionary containing experiment results
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract experiment class and remove it from config
    experiment_path = config.pop('experiment')
    
    # Run the experiment
    return run_experiment(experiment_path, config)


def compare_activations(
    experiment_class: Union[str, Type[BaseExperiment]],
    activations: List[str],
    base_config: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "results/activation_comparison",
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple activation functions using the same experiment setup.
    
    Args:
        experiment_class: Experiment class or dotted path to the class
        activations: List of activation function names to compare
        base_config: Base configuration for all experiments
        output_dir: Directory to save results
        **kwargs: Additional arguments to pass to all experiments
        
    Returns:
        Dictionary mapping activation names to their results
    """
    if base_config is None:
        base_config = {}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for activation in activations:
        # Update config for this activation
        config = base_config.copy()
        config.update({
            'activation': activation,
            'name': f"{config.get('name', 'experiment')}_{activation}",
            'root_dir': str(output_dir / activation),
            **kwargs
        })
        
        # Run the experiment
        print(f"\n{'=' * 80}")
        print(f"Running experiment with {activation} activation")
        print(f"{'=' * 80}")
        
        try:
            result = run_experiment(experiment_class, config)
            results[activation] = {
                'metrics': result.get('metrics', {}),
                'results': result.get('results', {})
            }
            
            # Save individual results
            result_file = output_dir / f"{activation}_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\nResults for {activation} saved to {result_file}")
            
        except Exception as e:
            logging.error(f"Error running experiment with {activation} activation: {str(e)}")
            results[activation] = {'error': str(e)}
    
    # Save comparison summary
    summary = {
        'activations': activations,
        'results': {
            act: {
                'test_accuracy': res.get('metrics', {}).get('test_accuracy', 0.0),
                'test_loss': res.get('metrics', {}).get('test_loss', 0.0),
                'training_time': res.get('metrics', {}).get('training_time', 0.0)
            }
            for act, res in results.items()
        }
    }
    
    summary_file = output_dir / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComparison summary saved to {summary_file}")
    
    return results


def generate_experiment_report(
    results: Dict[str, Any],
    output_file: Union[str, Path],
    format: str = 'markdown'
) -> None:
    """
    Generate a report from experiment results.
    
    Args:
        results: Experiment results dictionary
        output_file: Path to save the report
        format: Report format ('markdown' or 'html')
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'markdown':
        _generate_markdown_report(results, output_file)
    elif format == 'html':
        _generate_html_report(results, output_file)
    else:
        raise ValueError(f"Unsupported report format: {format}")


def _generate_markdown_report(results: Dict[str, Any], output_file: Path) -> None:
    """Generate a markdown report from experiment results."""
    with open(output_file, 'w') as f:
        f.write("# Experiment Report\n\n")
        
        # Metadata
        if 'metadata' in results:
            f.write("## Metadata\n\n")
            for key, value in results['metadata'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
        
        # Metrics
        if 'metrics' in results:
            f.write("## Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    value = f"{value:.4f}"
                f.write(f"| {key} | {value} |\n")
            f.write("\n")
        
        # Results
        if 'results' in results and results['results']:
            f.write("## Results\n\n")
            if isinstance(results['results'], dict):
                f.write("| Key | Value |\n")
                f.write("|-----|-------|\n")
                for key, value in results['results'].items():
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}"
                    f.write(f"| {key} | {value} |\n")
            f.write("\n")
        
        # Notes
        if 'notes' in results:
            f.write("## Notes\n\n")
            f.write(results['notes'] + "\n\n")
        
        # Timestamp
        import datetime
        f.write(f"\n*Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


def _generate_html_report(results: Dict[str, Any], output_file: Path) -> None:
    """Generate an HTML report from experiment results."""
    # Convert markdown to HTML using a simple template
    import tempfile
    import markdown
    
    # First generate markdown
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp:
        tmp_file = tmp.name
    
    try:
        _generate_markdown_report(results, tmp_file)
        
        # Convert markdown to HTML
        with open(tmp_file, 'r') as f:
            markdown_content = f.read()
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .timestamp {{ color: #666; font-style: italic; text-align: right; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""
        
        html_content = html_content.format(
            content=markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code']
            )
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
            
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file)
        except OSError:
            pass
