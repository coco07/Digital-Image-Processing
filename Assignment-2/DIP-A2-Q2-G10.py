import pandas as pd
import numpy as np
import cv2


def arithmetic_coding(sym_prob_mat, N, message="1"):
    """
    
    This function will encode a given message using a technique called arithmetic coding.
    sym_prob_mat : It's a (Nx2) matrix where N is the no. of unique symbols and the two columns are symbols and its probability.
    N : no. of unique source symbols.
    message : It's a 1-D array containing the message to be coded.
    
    """
    
    # our input "sym_prob_mat" can be either a matrix or an image, depends on the user and so let's convert image to the requ - 2 colm matrix

    # find the dimensions of our input,
    # if (columns == 2) then user has given 2 colm matrix of symbols and probabilities. no need to change it.
    # if (columns != 2) then user has given a single channel image as input. convert it to requ matrix format.
    image_matrix = np.array(sym_prob_mat)
    row,col = image_matrix.shape
    if col != 2 :
        # convert image to a single column matrix by reshaping
        sing_colm_img_mat = image_matrix.reshape(row*col,1)
        
        # in this case, our message to encode will be the intensity values, so making an array of all intensity values present in image
        message = []
        N=256
        for i in sing_colm_img_mat:
            message.append(i)
        
        # so in the case when input is an image, this will be our message to encode
        message = np.array(message)
        
        # we will need to add 256 new rows having values from 0 to 255, this will ensure that when we will use `value_counts()` to find frequency of each intensity value, we will not skip any intensity. ofcourse we will substract each intensity's frequency by 1 later.
        my= []
        for i in range (256):
            my.append(i)
        df_my = pd.DataFrame(my)
        
        # let's now convert our message into a pandas DataFrame in order to use the count_values(). 
        df_message = pd.DataFrame(message)
        new_df_message = df_message.append(df_my,ignore_index=True)

        # using value_counts() to get freq. of each intensity value and substracting 1 as mentioned above. 
        int_vs_freq = new_df_message.value_counts(sort=False,ascending=True)
        for i in range(256):
            int_vs_freq[i] = int_vs_freq[i]-1
        
        # let's convert our frequency values to corresponding proportions
        int_vs_prob = int_vs_freq / int_vs_freq.sum()
        mat_int_vs_prob = int_vs_prob.to_numpy()
        A = mat_int_vs_prob.reshape(256,1)
        
        # we got our probability column(A) and intensity column(B)
        B = np.array(range(256),dtype=int).reshape(256,1)
        
        # appending A and B to get our required 2 column matrix input
        requ_mat = np.append(A, B, axis=1)
        requ_mat[:, [1, 0]] = requ_mat[:, [0, 1]]
        sym_prob_mat = requ_mat.copy()

    # initiate some variables and lists as it will be required in code later
    j=1
    form = [0]
    to = []
    length = 1.0
    lt = []
    
    # we will loop through every letter in our message array.
    for sym in message:
        
        # converting the given matrix into a pandas DataFrame
        df1 = pd.DataFrame(sym_prob_mat)
        
        # renaming the columns, as converting from matrix to dataframe gave its column default names. Saving the new updated dataframe as 'df'
        df = df1.rename(columns = {0: 'symbol', 1: 'probability'}, inplace = False)
        
        # adding 3 more columns to the dataframe. Now our dataframe column names are symbol, probability, length, from & to.
        for i in range(N):
            lt.append(length * float(df.probability[i])) 
        for i in range(N-1):
            form.append(form[i]+float(lt[i]))
        for i in range(N):
            to.append(form[i]+float(lt[i]))
        df["length"] = lt
        df["from"] = form
        df["to"] = to
        
        # printing the dataframe table and the stage of iteration
        print(f"Stage_{j}")
        print(df)
        
        # updating out probability range based on message
        for i in range (N):
            if (df.symbol[i] == sym):
                a = form[i]
                form = []
                form.append(a)
                
                # printing the message or submessage and giving a range of probabilities. Any number lying in the given interval will be the corresponding message.
                print(f"\nMessage : {message[0:j]}  ::  Code : [{a} to {df.to[i]})\n")

                j=j+1
                to = []
                lt = []
                length = df.length[i]

def main():
    choice = int(input("Enter 1 for a single channel image compression or 2 for text compression: "))

    if choice == 1:
        D = cv2.imread('Q2_input_image.png', cv2.IMREAD_UNCHANGED)
        cv2.imwrite('grey_parrot.png',D[:,:,1])
        sym_prob_mat = cv2.imread('grey_parrot.png',cv2.IMREAD_UNCHANGED)
        N = 256
        arithmetic_coding(sym_prob_mat, N)

    elif choice == 2:
        sym_prob_mat = np.array([['s',0.0625],
                                 ['a',0.1875],
                                 ['r',0.0625],
                                 ['n',0.1875],
                                 ['g',0.0625],
                                 ['v',0.0625],
                                 ['i',0.1875],
                                 ['j',0.0625],
                                 ['y',0.0625],
                                 ['t',0.0625]])
        print(f"\nThe Symbol Probability Input Matrix : \n{sym_prob_mat}")
        N = 10
        print(f"\nNumber of unique symbols are : {N}")
        msg = ['s','a','r','a','n','g','v','i','j','a','y','n','i','t','i','n']
        print(f"\nMessage to Encode : {msg}\n")
        arithmetic_coding(sym_prob_mat, N, msg)

    else :
        print("Invalid choice")

if __name__ == '__main__':
    main()