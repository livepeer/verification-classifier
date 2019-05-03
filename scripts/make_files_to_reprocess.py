import argparse

parser = argparse.ArgumentParser(description='Generate renditions')

parser.add_argument('-i', "--input", action='store', help='Input file containing the bad processed renditions',
                    type=str, required=True)

args = parser.parse_args()
input_file = args.input


def get_files_to_reprocess(file_with_errors):
    attacks_and_files = {}
    with open(file_with_errors) as file_to_read:
        for current_line in file_to_read:
            processed_line = clean_string(current_line)
            for current_line in processed_line:
                key = get_key(current_line[0])
                if key in attacks_and_files:
                    attacks_and_files[key].add(current_line[1] + '\n')
                else:
                    attacks_and_files[key] = set()
    return attacks_and_files


def clean_string(string_to_clean):
    attack_and_file = []
    extension = '.mp4'
    files_to_process = [name + extension for name in string_to_clean.split('.mp4')]
    for line in files_to_process:
        split_line = line.split('/')
        if len(split_line) == 5:
            attack_and_file.append((split_line[3], split_line[4]))
    return attack_and_file


def get_key(full_key):
    sep = '_'
    split_key = full_key.split(sep)
    if len(split_key) == 1:
        return 'orig'
    return sep.join(split_key[1:])


def write_to_file(dict_to_write):
    for k, v in dict_to_write.items():
        with open(k + '_reprocess', 'w') as w_file:
            w_file.writelines(v)


if __name__ == "__main__":
    files_to_reprocess = get_files_to_reprocess(input_file)
    write_to_file(files_to_reprocess)
