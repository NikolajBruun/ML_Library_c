# ML_Library_c
Et primitivt ml bibliotek  som indeholder lineær regression logistisk regression og neuralt netværk


Der er 3 executables, nn-demo, LinReg-demo og LogReg-demo

nn-demo er et neuralt netværk med 1 hidden layer. Dataen er mnist (handwritten digits 0-9)

Input dim er 28x28 = 784 -> hidden layer har 128 neuroner og output lag selvfølgelig 10

Mnist dataen bliver normaliseret til [0,1], lr=0.1, og batchsize 100 (da det køres minibatch-sgd)

jeg har kun testet det med 1 epoke (dvs 
$$\frac{60,000}{100}=600$$
updates) men her er accuracy omkring 0.9
(det tager cirka samme tid som et game blitz chess)

Datasættet mnist_train var akkurat for stort til at blive uploadet til github -> jeg prøver at linke til et sted man kan downloade det


nn.c er en rodet omgang...
ideen er grundlæggende:
$$z^{(l)}=w^{(l)}h^{(l-1)}+b^{(l)}$$ 

og $$h^{(l)}=f(z^{(l)}) $$ -> hvor f er en activation function

For backward propagation er der derfor brug for at beregne 3 forskellige størrelser for hvert lag
$$\frac{\partial L}{\partial w}$$
$$\frac{\partial L}{\partial b}$$
$$\frac{\partial L}{\partial h}$$

og det er så alligevel heller ikke helt sandt for vi skal også udregne $$\frac{\partial L}{\partial \hat{y}}$$ fra vores output lag

I dette projekt bruger jeg cross-entropy-loss, og jeg bruger altid softmax i mit output lag
dvs vi kan bruge det trick der er $$\frac{\partial L}{\partial z^{(lastLayer)}}=\hat{y}-y$$ (nemmest at indse ved at udregne i hånden)

Der bliver ikke brugt softmax i andre lag, derfor er det egentlige afledte softmax ikke implementeret...

Relu er også brugt, den er ret simpelt differentieret, og bliver bare brugt elementvist på elementerne i z

Nu kommer den større ide generelt... dette var blot lidt notationsforklaring og hvorfor mine funktioner hedder gradient_w fx

for hvert lag bliver gradienten af vægt og bias udregnet og akkumuleret i en matrice, når *batchsize* inputs er kommet igennem opdatere vi vægtene (mini batch sgd)



HVAD KAN FORBEDRES:

1) Lige nu er main demo relativ lang, jeg gad godt at have main mere kompakt, og forhåbenligt vil det medføre et mere brugervenligt UI
2) hurtigere træning, evt med open_mp, det er ikke noget jeg har beskæftiget mig med overhovedet. Problemet er at hvert iteration i træningsloopet tilgår de samme matricer (gradienterne til vægtmatricerne og biasvektoren) at blot lave det yderste træningsloop om vil resultere i race condition.... nederen
3) DOKUMENTATION!!!


logreg er meget simpel og forståelig, god genopfriskning for teorien, mere interresant er den ikke. Dog er demoen meget sjov, med meget lidt træning score den rigtig fint på wisconsin breast cancer datasæt (acc 0.98) til klassificering om godartet/ondartet kraftknude, god læring at implementere gradient descent her inden backprop

linreg er kedelig, demoen predicter huspriser, udfra boston housing dataset... dog har vejen til linreg været lang, det var den første ML model og krævede derfor at jeg byggede hele matrix biblioteket, linreg bruger analytiske løsning, og derfor har jeg også implementeret naiv-matrix invers (gauss-jordan pivot alt det der)






