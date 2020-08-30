# zJavaAI
Simple, adaptable, fully connected neural network with built-in image support.<br>
Compile with <a href="https://github.com/fommil/matrix-toolkits-java">mtj-1.0</a> and <a href="https://math.nist.gov/javanumerics/jama/">Jama 1.0.3</a>.
<br><br>
Usage:
1. Enter Interface.java, change FILE_NAME_STRING to the directory holding your training data.<br>
2. Set AI_NAME to the name of the file storing the neural net. this file can get large.<br>
3. Set Bx and By to the desired image size, and MONOCHROME to true or false<br>
4. set TRAIN to true while training. then to false to simply view generated images.<br>
5. Set up other metadata as necessary or desired.

<br>
The neural net can function with other data types other than images, but you first need functions that convert your data to and from nVector's.<br>
The <b>NeuralNetHelper</b> class contains a straightforward framework to construct a neural network for any data type.<br>
When using the neural net on non-image data types, it is recommended to not use the Interface class.
