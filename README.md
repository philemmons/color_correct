# color_correct algoritm

Title: Color-Correct, it is a work in progress. 
Copy image into main4.py file location, convert to a jpg, and rename it '1a.jpg'

Abstract: Image is altered to and from a specific color blindness. 
The algorithm used to simulate color blindness is based upon a color matrix found here:
http://web.archive.org/web/20081014161121/http://www.colorjack.com/labs/colormatrix/
This author color matrix is based on Daltonize.org. 

The general idea is re-create a RGB color channel mixer at the pixel level. Convert the image to 
a color blind image and reverse the process.An image is split into its RGB band. From there, 
convert the single band to a RGB, three bands, and split them again. 
Adjustments are made to each image band as per the color matrix. The single band image are layered,
with transparency as a mask, thanks to the female TA for suggesting this. Next, the image's brightness 
levels are raised/lowered as to account for darken results from the color matrix. 

Testing the images for correctness was performed by using a color blind simulator.
http://www.color-blindness.com/coblis-color-blindness-simulator/

The color blind converter works on blue-blind and red-blind. As for the other two, green is to dominate.
Monochromacy is replaced with an opencv gray scale built in feature.

All of the adjustments DO NOT work, yet. The reciprocal of the color matrix is applied to the color blind images, 
but the results are out of range of 255. First approach would be to normalize the images by each bands. 
Second, change the sequence of layering based on the color blindness instead of universal layering.
Additional work would include a dampening for the green color in the green-blind and monochromacy. Perhaps
a various color matrix from daltonize.org would do. Although there are Daltonize tranformation matrices.

As suggested by vischeck.com, the quick fix is to increase the red and green contrast as to make them more distinct. 
Also, they offer another approach that is to, "convert these (color blind type) into changes in brightness, 
and blue/yellow coloration." This would give a new color depth which would help identify the color.
 

Author: Phillip T. Emmons
Alas, outside of a few explanations from Dylan and Chris, this was written by me.

Date: 10.14.2016

Protanopia:{ R:[56.667, 43.333, 0], G:[55.833, 44.167, 0], B:[0, 24.167, 75.833]}
Deuteranopia:{ R:[62.5, 37.5, 0], G:[70, 30, 0], B:[0, 30, 70]}
Tritanopia:{ R:[95, 5, 0], G:[0, 43.333, 56.667], B:[0, 47.5, 52.5]}
Achromatopsia:{ R:[29.9, 58.7, 11.4], G:[29.9, 58.7, 11.4], B:[29.9, 58.7, 11.4]}

