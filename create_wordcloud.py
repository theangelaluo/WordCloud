import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

"""
Before you run this program, make sure you run the following commands in your Terminal:

pip3 install numpy
pip3 install wordcloud
pip3 install matplotlib

Source: https://www.datacamp.com/community/tutorials/wordcloud-python

"""


ALL_FILES = ['blm.txt', 'blm1.txt', 'blm2.txt', 'blm3.txt', 'blm4.txt', 'blm5.txt']
WORDS_TO_IGNORE = set(STOPWORDS)
IMAGE_FILE = 'fist.png'


def main():
    text_data = ''
    for filename in ALL_FILES:
        text_data += add_text_from_file(filename)

    #SIMPLE VERSION
    #wordcloud = WordCloud(stopwords=WORDS_TO_IGNORE, max_words=200, max_font_size=90).generate(text_data)

    #VERSION WITH IMAGE
    mask = create_image_mask()
    wordcloud = WordCloud(background_color='black', contour_width=3, contour_color='white', mask=mask, stopwords=WORDS_TO_IGNORE, max_words=150, max_font_size=100).generate(text_data)
    wordcloud.to_file('wordcloud.png')

    #Plot the WordCloud    
    plt.figure(figsize=[8,8])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()



def add_text_from_file(filename):
    text_to_add = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            text_to_add += line
    return text_to_add


def create_image_mask():
    """
    Creates and formats a "mask" out of the image file to use as the background for the word cloud.
    """
    fist = np.array(Image.open(IMAGE_FILE))
    transformed_fist = np.ndarray((fist.shape[0],fist.shape[1]), np.int32)
    for i in range(len(fist)):
        transformed_fist[i] = list(map(transform_format, fist[i]))
    return transformed_fist



def transform_format(val):
    """
    Turns dark/black pixels to white so that the image has the correct format to act as a mask.
    """
    if val == 0: #change this to 2 if you want the words inside the fist, instead of in the background
        return 255
    else:
        return val



if __name__ == '__main__':
    main()
