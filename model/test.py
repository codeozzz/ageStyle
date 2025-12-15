import os
import sys
import argparse
import importlib
import yaml

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
from os.path import join as pjoin

from data_loader import process_single_bvh
from trainer import Trainer
from remove_fs import remove_fs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AgeStyle Motion Style Transfer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Transfer neutral walk to childlike style
    python test.py --content_src data/xia_test/neutral_01_000.bvh \\
                   --style_src data/xia_test/childlike_01_000.bvh
    
    # Transfer old walk to sexy style
    python test.py --content_src data/xia_test/old_01_000.bvh \\
                   --style_src data/xia_test/sexy_01_001.bvh \\
                   --output_dir my_output
        """
    )
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: pretrained)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--config', type=str, default='config',
                        help='Config module name')
    parser.add_argument('--content_src', type=str, 
                        default=pjoin(BASEPATH, "data/xia_test/old_18_000.bvh"),
                        help='Path to content BVH file (motion to transform)')
    parser.add_argument('--style_src', type=str,
                        default=pjoin(BASEPATH, "data/xia_test/sexy_22_000.bvh"),
                        help='Path to style BVH file (target style)')
    parser.add_argument('--output_dir', type=str, default="output",
                        help='Output directory for generated BVH files')

    return parser.parse_args()


def main(args):
    """
    Main function for motion style transfer.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Load configuration
    config_module = importlib.import_module(args.config)
    config = config_module.Config()
    config.initialize(args, save=False)

    # Initialize trainer and load pretrained model
    print("Loading model...")
    trainer = Trainer(config)
    trainer.to(config.device)
    trainer.resume()
    
    # Process input BVH files
    print(f"Processing content: {args.content_src}")
    co_data = process_single_bvh(args.content_src, config, to_batch=True)
    
    print(f"Processing style: {args.style_src}")
    st_data = process_single_bvh(args.style_src, config, to_batch=True)

    # Run inference
    print("Running style transfer...")
    output = trainer.test(co_data, st_data, '3d')
    
    # Extract output
    foot_contact = output["foot_contact"][0].cpu().numpy()
    motion = output["trans"][0].detach().cpu().numpy()
    
    # Determine output directory
    output_dir = pjoin(config.main_dir, 'test_output') if args.output_dir is None else args.output_dir
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get label names for output filename
    dataset_config = pjoin(BASEPATH, "global_info/xia_dataset.yml")
    with open(dataset_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    content_names = cfg["content_names"]
    style_names = cfg["style_names"]

    # Get labels from data
    input_content_content = content_names[co_data["content_label"]]
    input_content_style = style_names[co_data["label"]]
    input_style_content = content_names[st_data["content_label"]]
    input_style_style = style_names[st_data["label"]]

    # Generate output filename
    output_filename = "{}_{}_by_{}_{}.bvh".format(
        input_content_style, 
        input_content_content, 
        input_style_style, 
        input_style_content
    )
    output_path = pjoin(output_dir, output_filename)
    
    # Save output
    print(f"Saving output to: {output_path}")
    remove_fs(motion, foot_contact, output_path=output_path)
    
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
