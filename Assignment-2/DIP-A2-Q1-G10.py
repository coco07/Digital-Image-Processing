import re
import numpy as np
from PIL import Image, ImageOps


def huffman(my_string, choice, message_to_encode=[]):
    if choice == 1:
        shape = my_string.shape
        a = my_string
        my_string = str(my_string.tolist())     # converting image matrix to list of characters

        letters = []
        only_letters = []
        prob_table = {}
        for letter in my_string:
            if letter not in letters:
                frequency = round(my_string.count(letter)/len(my_string), 2)  # frequency normalisation for each letter
                letters.append(frequency)
                letters.append(letter)
                only_letters.append(letter)
                prob_table[letter] = frequency

        print(only_letters)
        print(f"\nProbability table used: {sorted(prob_table.items())}")

        nodes = []
        while len(letters) > 0:
            nodes.append(letters[0:2])
            letters = letters[2:]  # sorting according to frequency
        nodes.sort()
        huffman_tree = []
        huffman_tree.append(nodes)  # Make each unique character as a leaf node

        def combine_nodes(nodes):
            pos = 0
            newnode = []
            if len(nodes) > 1:
                nodes.sort()
                nodes[pos].append("0")  # assigning values 1 and 0
                nodes[pos + 1].append("1")
                combined_node1 = (nodes[pos][0] + nodes[pos + 1][0])
                combined_node2 = (nodes[pos][1] + nodes[pos + 1][1])  # combining the nodes to generate pathways
                newnode.append(combined_node1)
                newnode.append(combined_node2)
                newnodes = []
                newnodes.append(newnode)
                newnodes = newnodes + nodes[2:]
                nodes = newnodes
                huffman_tree.append(nodes)
                combine_nodes(nodes)
            return huffman_tree  # huffman tree generation

        newnodes = combine_nodes(nodes)

        huffman_tree.sort(reverse=True)

        checklist = []
        for level in huffman_tree:
            for node in level:
                if node not in checklist:
                    checklist.append(node)
                else:
                    level.remove(node)

        # print("Huffman tree generated:")
        # count = 0
        # for level in huffman_tree:
        #     print("Level", count, ":", level)  # print huffman tree
        #     count += 1

        letter_binary = []
        if len(only_letters) == 1:
            lettercode = [only_letters[0], "0"]
            letter_binary.append(lettercode * len(my_string))
        else:
            for letter in only_letters:
                code = ""
                for node in checklist:
                    if len(node) > 2 and letter in node[1]:  # generating binary code
                        code = code + node[2]
                lettercode = [letter, code]
                letter_binary.append(lettercode)

        print("\ncode book generated:")
        print("Symbol\tcode-word")
        for letter in letter_binary:
            print(f" {letter[0]}\t\t {letter[1]}")

        bitstring = ""
        for character in my_string:     # compressed image generation
            for item in letter_binary:
                if character in item:
                    bitstring = bitstring + item[1]
        binary = "0b" + bitstring

        output = open("Q1_compressed.txt", "w+")
        output.write(bitstring)

        print("\nCompressed file generated as compressed.txt")

        print("\nStarted Decoding.......")
        bitstring = str(binary[2:])
        uncompressed_string = ""
        code = ""
        for digit in bitstring:     # uncompressed image generation
            code = code + digit
            pos = 0  # iterating and decoding
            for letter in letter_binary:
                if code == letter[1]:
                    uncompressed_string = uncompressed_string + letter_binary[pos][0]
                    code = ""
                pos += 1

        temp = re.findall(r'\d+', uncompressed_string)
        res = list(map(int, temp))
        res = np.array(res)
        res = res.astype(np.uint8)
        res = np.reshape(res, shape)
        print("Input image dimensions:", shape)
        print("Output image dimensions:", res.shape)
        data = Image.fromarray(res)
        data.save('Q1_output_image.jpg')     # uncompressed image as an output
        if a.all() == res.all():
            print("Decoded successfully")

    else:
        from collections import defaultdict
        from heapq import heappush, heappop, heapify

        print(f"\nProbability table used: {my_string}\n")

        code = defaultdict(list)    # mapping of letters to codes

        # Using a heap makes it easy to pull items with lowest frequency. Items in the heap are tuples containing a list of letters and the combined frequencies of the letters in the list.
        heap = [ ( freq, [ ltr ] ) for ltr,freq in my_string.items() ]
        heapify(heap)
        # Reduce the heap to a single item by combining the two items with the lowest frequencies.
        while len(heap) > 1:
            freq0,letters0 = heappop(heap)
            for ltr in letters0:
                code[ltr].insert(0,'0')

            freq1,letters1 = heappop(heap)
            for ltr in letters1:
                code[ltr].insert(0,'1')

            heappush(heap, ( freq0+freq1, letters0+letters1))
            print(f"merging lowest probability pair {letters0+letters1}  ======>  {heap}")
        print()
        for k,v in code.items():
            code[k] = ''.join(v)

        encoded_message = ''
        for i in message_to_encode:
            encoded_message += code[i]
        print(f"encoded_message for {message_to_encode}  ======>  {encoded_message}")

        return code

def main():
    choice = int(input("Enter 1 for a single channel image compression or 2 for text compression: "))

    if choice == 1:             # image compression
        my_string = Image.open("Q1_input_image.jpg")       # expects image to be present at the same place that of .py file
        my_string = np.asarray(ImageOps.grayscale(my_string), np.uint8)
        huffman(my_string, choice)

    elif choice == 2:           # text compression
        my_string = {'a': 0.25, 'b': 0.25, 'c': 0.2, 'd': 0.15, 'e': 0.15}
        message_to_encode = ['a','b','a','c']
        code = huffman(my_string, choice, message_to_encode)

        print("\ncode book generated:")
        print("Symbol\tcode-word")
        for k, v in sorted(code.items()):
            print(f" {k}\t\t {v}")

    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()