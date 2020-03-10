import numpy as np
import argparse
import os


class Node:
    def __init__(self, value):
        self.diag = None
        self.up = None
        self.left = None
        self.value = value


def build_alignment_tree(seq1, seq2, similar_score=1, gap_penalty=1, mismatch_penalty=1):
    assert isinstance(seq1, str) and isinstance(seq2, str), 'Sequences are not strings'
    score_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                current_similarity = similar_score
            else:
                current_similarity = mismatch_penalty * -1
            score_matrix[i, j] = max(
                score_matrix[i - 1, j - 1] + current_similarity,
                score_matrix[i - 1, j] - gap_penalty,
                score_matrix[i, j - 1] - gap_penalty,
                0
            )
    start_position = tuple(
        np.unravel_index(
            np.argmax(score_matrix), shape=score_matrix.shape
        )
    )
    root = Node(start_position)
    current_positions = [root]
    while len(current_positions) > 0:
        next_positions = list()
        for node in current_positions:
            if seq1[node.value[0] - 1] == seq2[node.value[1] - 1]:
                current_similarity = similar_score
            else:
                current_similarity = mismatch_penalty * -1
            for step_score, coordinates, direction in \
                    (current_similarity, (node.value[0] - 1, node.value[1] - 1), 'diag'), \
                    (-1 * gap_penalty, (node.value[0] - 1, node.value[1]), 'up'), \
                    (-1 * gap_penalty, (node.value[0], node.value[1] - 1), 'left'):
                if coordinates[0] > 0 and \
                        coordinates[1] > 0 and \
                        score_matrix[coordinates[0], coordinates[1]] > 0 and \
                        score_matrix[coordinates[0], coordinates[1]] + step_score == \
                        score_matrix[node.value[0], node.value[1]]:
                    setattr(node, direction, Node(value=coordinates))
                    next_positions.append(getattr(node, direction))
        current_positions = next_positions
    return root, score_matrix


def extract_alignments_recursive(root, final_paths, prefix=None):
    if prefix is None:
        prefix = list()
    prefix.append(root.value)
    for direction in root.left, root.up, root.diag:
        leaf = True
        if direction is not None:
            leaf = False
            extract_alignments_recursive(direction, final_paths, prefix.copy())
    if leaf:
        final_paths.append(prefix)


def extract_alignments_from_root(root):
    result = list()
    extract_alignments_recursive(root, result)
    return result


def alignment_path_to_seqs(alignment_path, seq1, seq2):
    result = [[], []]
    for i in range(len(alignment_path) - 1):
        if alignment_path[i] == (alignment_path[i + 1][0] + 1, alignment_path[i + 1][1] + 1):
            result[0].append(seq1[alignment_path[i][0] - 1])
            result[1].append(seq2[alignment_path[i][1] - 1])
        elif alignment_path[i] == (alignment_path[i + 1][0] + 1, alignment_path[i + 1][1]):
            result[0].append(seq1[alignment_path[i][0] - 1])
            result[1].append('-')
        elif alignment_path[i] == (alignment_path[i + 1][0], alignment_path[i + 1][1] + 1):
            result[0].append('-')
            result[1].append(seq2[alignment_path[i][1] - 1])
        else:
            raise ValueError('Invalid alignment path: {p}'.format(p=alignment_path))
    result[0].append(seq1[alignment_path[-1][0] - 1])
    result[1].append(seq2[alignment_path[-1][1] - 1])
    return result


def align(seq1, seq2, similar_score=1, gap_penalty=1, mismatch_penalty=1, print_info=True):
    alignment_tree_root, score_matrix = build_alignment_tree(
        seq1, seq2, similar_score=similar_score, gap_penalty=gap_penalty, mismatch_penalty=mismatch_penalty
    )
    alignments_coordinates = extract_alignments_from_root(alignment_tree_root)
    alignments = [alignment_path_to_seqs(item, seq1, seq2) for item in alignments_coordinates]

    if print_info:
        print('Sequence 1:', seq1)
        print('Sequence 2:', seq2)
        row_names = ' ' + seq1
        print(
            'Score matrix:',
            '\t\t' + '\t'.join(seq2),
            *(row_names[i] + '\t' + '\t'.join(map(str, score_matrix[i, :])) for i in range(score_matrix.shape[0])),
            sep='\n'
        )
        print('Best alignment score: {s}'.format(s=np.max(score_matrix)))
        for alignment_path, alignment in zip(alignments_coordinates, alignments):
            print('\nAlignment:', ''.join(alignment[0][::-1]), ''.join(alignment[1][::-1]), sep='\n')
            path_matrix = np.full(score_matrix.shape, ' ', dtype='U')
            for i in range(len(alignment_path) - 1):
                if alignment_path[i] == (alignment_path[i + 1][0] + 1, alignment_path[i + 1][1] + 1):
                    path_matrix[alignment_path[i][0], alignment_path[i][1]] = '\\'
                elif alignment_path[i] == (alignment_path[i + 1][0] + 1, alignment_path[i + 1][1]):
                    path_matrix[alignment_path[i][0], alignment_path[i][1]] = '|'
                elif alignment_path[i] == (alignment_path[i + 1][0], alignment_path[i + 1][1] + 1):
                    path_matrix[alignment_path[i][0], alignment_path[i][1]] = '-'
            path_matrix[alignment_path[-1][0], alignment_path[-1][1]] = '\\'
            print('Alignment path:')
            print(
                '  |   | ' + ' | '.join(seq2) + ' |',
                *(' | '.join((row_names[i], *path_matrix[i, :])) + ' |' for i in range(path_matrix.shape[0])),
                sep='\n' + '---' + '----' * (path_matrix.shape[1]) + '\n'
            )
    return alignments


def main(args):
    seqs = [args.seq1, args.seq2]
    for i, attribute in zip(range(2), ('seq1', 'seq2')):
        if os.path.exists(getattr(args, attribute)):
            with open(getattr(args, attribute)) as input_d:
                seqs[i] = input_d.read()
    return align(
        seq1=seqs[0],
        seq2=seqs[1],
        gap_penalty=args.gap_penalty,
        mismatch_penalty=args.mismatch_penalty,
        similar_score=args.similar_score
    )


if __name__ == '__main__':
    examples = """Examples: 
    python3 smith_waterman.py --seq1 AAAA --seq2 AAAA
    python3 smith_waterman.py --seq1 file.txt --seq2 AAAA
    """
    parser = argparse.ArgumentParser(
        description='Smith-Waterman algorithm implementation',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--seq1', required=True, help='path to .txt file with sequence or sequence string')
    parser.add_argument('--seq2', required=True, help='path to .txt file with sequence or sequence string')
    parser.add_argument('--similar_score', default=1, required=False)
    parser.add_argument('--gap_penalty', default=1, required=False)
    parser.add_argument('--mismatch_penalty', default=1, required=False)
    main(parser.parse_args())
