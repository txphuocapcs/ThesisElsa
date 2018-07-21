from Bio import pairwise2
from Bio.pairwise2 import format_alignment
#for a in pairwise2.align.localms(["A", "C", "C", "G", "T", "N97", "C", "T"], ["A", "C", "C", "G", "8DX", "C", "T"], 2, -1, -.5, -.1, gap_char=["-"]):
    #print(format_alignment(*a))

def algn(seq1, seq2):
    testseq=["A", "C", "C", "G", "T", "N97", "C", "T"]
    for i in range(seq1.size):
        seq1[i]=''.join(j for j in seq1[i] if not j.isdigit())
    return pairwise2.align.localms(seq1.tolist(), seq2.split(), 2, -1, -.5, -.1, gap_char=["-"])