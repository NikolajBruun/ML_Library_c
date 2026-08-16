# ML_Library_c
Et primitivt ml bibliotek  som indeholder lineær regression logistisk regression og neuralt netværk


Der er 3 executables, nn-demo, LinReg-demo og LogReg-demo

nn-demo er et neuralt netværk med 1 hidden layer. Dataen er mnist (handwritten digits 0-9)
Input dim er 28x28 = 784 -> hidden layer har 128 neuroner og output lag selvfølgelig 10
Hidden layer bruger relu activationfunction og output softmax
Backpropagation er 
$$\frac{1}{2}$$
