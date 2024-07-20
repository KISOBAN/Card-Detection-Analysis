In order to determine the Attribute:
1.) First the card was cropped out so that only the image of the attribute remained. 
2.) Some Preprocessing of the input image was required so that when we threshold the image only the text and symbol would be white. This was done by lowering the contrast and slightly increasing the brightness so that no unneccesary white pixels would occur
3.) the input image was gray scaled and thresholded with a binary threshold
4.) The thresholded input image was then compared to the stock threshold images (found in the 'ThresholdedStockImages' Folder) with an AND operation so that any pixels that do not overlap with each other are blacked out leaving only the original attribute of the card
5.) The resulting image that had the highest pixel count was chosen and that determined what attribute the image was

It can be seen that when the input image is ANDED with the earth attribute it retains most of its white pixels so therefore this card is an earth attribute
