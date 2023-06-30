import sys

if len(sys.argv) != 8:
    sys.exit("Incorrect command format, sample command: python3 execute.py image image.txt 0.1 VA 0.5 prune output.txt")

file_type, filename = sys.argv[1], sys.argv[2]
file = open(filename, 'r')
binary_threshold, fairing_times, pruning_method, pruning_threshold, output_type, output_filename = \
    None, None, None, None, None, None

if file_type == "image":
    if 0 <= float(sys.argv[3]) <= 1:
        binary_threshold = sys.argv[3]
    else:
        sys.exit("Invalid binary threshold, valid threshold is between 0 and 1")
elif file_type == "curve":
    if int(sys.argv[3]) >= 0:
        fairing_times = sys.argv[3]
    else:
        sys.exit("Invalid fairing number, valid fairing number is a non-negative integer")
else:
    sys.exit("Invalid file type, valid types are image and curve")

if sys.argv[4] == "VA" or sys.argv[4] == "ET":
    pruning_method = sys.argv[4]
else:
    sys.exit("Invalid pruning method, valid methods are VA and ET")

if 0 <= float(sys.argv[5]) <= 1:
    pruning_threshold = sys.argv[5]
else:
    sys.exit("Invalid pruning threshold, valid threshold is between 0 and 1")

if sys.argv[6] == "color" or sys.argv[6] == "prune":
    output_type = sys.argv[6]
else:
    sys.exit("Invalid output type, valid types are color and prune")

output_filename = sys.argv[7]

# Input file is an image
