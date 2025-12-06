from argparse import ArgumentParser

# Add Help when finished
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--run_name', type=str, default='', help='', 
    )   
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=2, help='Increased to reduce memory usage',
    )
    parser.add_argument(
        '--project_name', type=str, default='DocRes', help='',
    )
    parser.add_argument(
        '--seed', type=int, default=0, help='',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Reduced to prevent OOM errors'
    )
    
    parser.add_argument(
        '--alpha', default=1e-2, type=float, help=''
    )
    
    parser.add_argument(
        '--image_size', type=int, default=288, help='', 
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4, help='', 
    )
    parser.add_argument(
        '--warmup_steps', type=int, default=1000, help='', 
    )
    parser.add_argument(
        '--num_epochs', type=int, default=10, help='', 
    )
    parser.add_argument(
        '--log_freq', type=int, default=1, help='', 
    )
    parser.add_argument(
        '--save_freq', type=int, default=100, help='', 
    )
    parser.add_argument(
        '--eval_bz', type=int, default=256, help='', 
    )
    parser.add_argument(
        '--output_dir', type=str, default='logs', help='', 
    )
    parser.add_argument(
        '--dataset_name', type=str, default='./data/uds-rift-cat-munchkin-250914', help='', 
    )
    parser.add_argument(
        '--gpu_ids', type=str, default='0,1,2,3', help='',
    )
    
    parser.add_argument(
        '--do_eval', action='store_true', help='Run evaluation after training',
    )
    
    #bool is "action='store_true'"
    return parser.parse_args()