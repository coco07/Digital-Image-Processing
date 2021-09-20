# Digital_Image_Processing

Assignment-I

Question-I:

Description:

Here we compute the shortest path between two pixels using three different distance measures, namely 4-adjacency, 8-adjacency, and m-adjacency measures.
The pixels are first mapped into undirected graphs where each pixels is a graph node and the connections are formed based on the type of adjacency measure selected.
Thereafter, the shortest path between two pixels is found using Breadth First search algorithms. Other algorithms that can also be used for finding the shortest path are Djikstra's, A*, etc.


Dependencies used:
 
#numpy: to load the image segment 
#opencv(cv2): needed only if image needs to be imported.


How To Run the program?

To load an image or change the image segment, head over down to the main function block and change the value of I.

To change the start and end locations of the pixels, again, change p and q values (location of the two pixels) respectively in the main function block. Same goes for changing the predefined set V and path_type of your interest. After heading to the directory of the program, following is the command to run the program in terminal:

> python Pixel_Digital_Path.py

the output will be printed in the same terminal.


-----------------------------------------------------------------


Question-II:

Description:

Here we are taking (M + 2*(border) * N + 2*(border)) 2D matrix to represent grayscale image along with it's border. We are trying to draw n number of rectangles inside this image, which must be non-overlapping with each other. In case if there is no space for the new rectangle, instead of throwing recursionError exception, program calls itself recursively to make canvas size double in order to fit n number of rectangles. Also along with the output image, we are printing final shape of the image.


Dependencies used:
 
#numpy: to load the image segment 
#opencv(cv2): to show/save final image.


How To Run the program?

To modify parameters, head over to the main function block and change the respective value of the variable.
Here, As Vf and Vb are optional parameters, if not provided function will still run without any issue but resulting into a binary image. After heading to the directory of the program, following is the command to run the program in terminal:

> python Create_Rectangles.py

this will save the output final_image.jpg into the same location where the program is stored. Please open the saved file to verify the output.

=================================================================================================================================================
