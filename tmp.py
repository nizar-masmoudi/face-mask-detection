from dataset.facemask import FaceMask
import argparse

def main():
  parser = argparse.ArgumentParser(description = 'Delete me plzz')
  parser.add_argument('-o', '--output', help = 'No.', required = True)
  args = parser.parse_args()
  FaceMask.download(args.output)

if __name__ == '__main__':
  main()