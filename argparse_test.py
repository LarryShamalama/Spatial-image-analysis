from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--file_name',
                    help='Image on which Bayesian model will be fitted on', 
                    type=str)

args = parser.parse_args()
print(args.file_name)