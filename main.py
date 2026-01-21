from leica_converter import convert_leica
import sys
import argparse

parser = argparse.ArgumentParser(description='Convert Leica files')
parser.add_argument('--inputfile', required=True, help='Path to the input LIF/LOF/XLEF file')
parser.add_argument('--image_uuid', default='n/a')
parser.add_argument('--show_progress', action='store_true')
parser.add_argument('--outputfolder', default=None, required=True,)
parser.add_argument('--altoutputfolder', default=None)
parser.add_argument('--tempfolder', default=None, help='Custom temp folder for intermediate files. If not set, uses system temp directory.')
parser.add_argument('--xy_check_value', type=int, default=3192)
parser.add_argument('--get_image_metadata', action='store_true', help='Include full image metadata JSON in keyvalues.image_metadata_json')
parser.add_argument('--get_image_xml', action='store_true', help='Include raw image XML in keyvalues.image_xml when available')

args = parser.parse_args()

result = convert_leica(
    inputfile=args.inputfile,
    image_uuid=args.image_uuid,
    show_progress=args.show_progress,
    outputfolder=args.outputfolder,
    altoutputfolder=args.altoutputfolder,
    tempfolder=args.tempfolder,
    xy_check_value=args.xy_check_value,
    get_image_metadata=args.get_image_metadata,
    get_image_xml=args.get_image_xml,
)

if result and result != "[]":
    print(result)
    sys.exit(0)
else:
    print("Error")
    sys.exit(1)

