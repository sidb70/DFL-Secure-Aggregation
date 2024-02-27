import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Start simulation server')
    parser.add_argument('--nodes', type=int, default=10, help='Number of nodes in DFL network')
    args = parser.parse_args()
    print(f'Starting simulation with {args.nodes} nodes')
    # Start server