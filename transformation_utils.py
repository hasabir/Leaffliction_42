import argparse


def parse_arguments(help=False):
    parser = argparse.ArgumentParser(
                description="Apply computer vision" +
                "transformations to extract features from leaf images.",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Examples:
  Display transformations for a single image:
    %(prog)s path/to/image.jpg

  Batch process a directory:
    %(prog)s -src path/to/images/ -dst path/to/output/
        """
    )

    parser.add_argument(
        'file',
        nargs='?',
        help='Path to a single image file (displays transformations)'
    )
    parser.add_argument(
        '--src',
        type=str,
        help='Source directory containing images'
    )
    parser.add_argument(
        '--dst',
        type=str,
        help='Destination directory to save transformations'
    )
    args = parser.parse_args()
    if help:
        parser.print_help()
        exit(0)
    return args
