from dataset.facemask import FaceMask
import argparse

def main():
  parser = argparse.ArgumentParser(description = 'Download and setup dataset')
  parser.add_argument('-o', '--output', help = 'Output folder', default = 'data')
  args = parser.parse_args()
  FaceMask.download(args.output)

if __name__ == '__main__':
  main()