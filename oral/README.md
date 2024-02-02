# Response to almost all qestions you must know the answers to before attending the exam (questions taken from Galatolo's repository) 

## a. Autograd and SGD

- **a.01** What is the principle behind using Stochastic Gradient Descent (SGD) for optimizing a function w.r.t. its parameters? <br>
    **Response** Dato che il costo computazionale dell'algoritmo del Gradiente Descendente è O(m), man mano che le dimensioni del set di addestramento crescono, il tempo necessario per compiere un singolo passo di gradiente diventa proibitivamente lungo, e quindi si utilizza lo SGD. **L'idea alla base del gradiente stocastico è che il gradiente è un'exepctation.** Nello specifico, ad ogni passo dell'algoritmo, possiamo estrarre casualmente un minibatch di esempi di dimensione m' dal set di addestramento. La stima del gradiente viene formata come segue:
$$\[\hat{g} = \frac{1}{m'} \sum_{i \in m'} \nabla J(\theta; x^{(i)}, y^{(i)})\]$$
L'algoritmo del gradiente stocastico poi segue il gradiente stimato verso il basso:
$$\[\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t; x^{(i)}, y^{(i)})\]$$
Dove eta è il tasso di apprendimento.
L'algoritmo di ottimizzazione non garantisce di raggiungere un minimo locale in un tempo ragionevole, ma spesso trova un valore molto basso della funzione di costo abbastanza rapidamente da essere utile. È il principale modo per addestrare modelli lineari su set di dati molto grandi.

- **a.02** Explain the concept of function composition in neural networks and how it relates to layers in a model.<br>
    **Response**: Il concetto di composizione delle funzioni nelle reti neurali si riferisce alla pratica di combinare più strati (layers) per formare una rete neurale complessa. In una rete neurale, ogni strato è costituito da neuroni e ognuno di questi strati esegue una trasformazione matematica sui dati in input. La composizione di queste trasformazioni attraverso gli strati della rete rappresenta la funzione complessiva che la rete sta apprendendo. Quando diciamo "composizione delle funzioni" in questo contesto, intendiamo che l'output di uno strato diventa l'input per il successivo. Quindi, se hai uno strato f1 seguito da uno strato f2, la composizione di f1 e f2 è f2(f1(x)), dove x è l'input originale.
  Ad esempio, se x rappresenta l'input di una rete neurale, l'output dello strato f1 sarà f1(x), e l'output totale della rete dopo la composizione con f2 sarà f2(f1(x)). Questa composizione di strati permette alla rete neurale di apprendere rappresentazioni più complesse e astratte dei dati. Ogni strato può catturare diversi aspetti o feature dei dati di input, e la composizione di più strati consente alla rete di apprendere rappresentazioni sempre più sofisticate man mano che si sposta attraverso gli strati.

- **a.03 / a.04** How do you calculate the derivative of a composed function w.r.t. its inputs? Describe the unfolding process of a function and its impact on derivative calculation.<br>
    **Response**: Usando la **chain rule**. Cioè usando la moltiplicazione delle derivate parziali (unfolding)
    $$\[\frac{do}{di} = \frac{do}{dw_2} \cdot \frac{dw_2}{dw_1} \cdot \frac{dw_1}{di}\]$$

    Dove:
    - o è l'output.
    - i è l'input.
    - w1 è la funzione più interna
    - w2 la funzione che prende l'output di w1.
    
    L'effetto dell'unfolding è quello di semplificare il calcolo delle derivate, specialmente in contesti in cui le funzioni coinvolte sono intricate.

- **a.06** Explain the derivation process for a function involving a binary operator and how it splits the derivation flow.<br>
    **Response:** In R ogni operatore binario chiuso (4 operazioni di base) creano una somma di due flussi di derivate (pensa alla derivata della somma o del prodotto ad esempio).
- **a.07** What is reverse mode differentiation, and how is it applied to compute derivatives in computational graphs?<br>
    **Response**: Per calcolare la derivata dell'output w.r.t. ad un input devi trovare tutti i possibili percorsi dall'output all'input e moltiplicare i valori parziali su ciascun percorso (ogni percorso corrisponde a una composizione di funzioni) e sommare tutti i risultati (ogni split rappresenta la presenz adi un operatore binario).
    **Nel reverse mode si calcola la derivata parziale di *un output* rispetto ad *ogni input* in passaggio**

- **a.08** Discuss the concept of forward path and backward path in the context of computational graphs and differentiation.<br>
    **Response**: Funziona all'esatto opposto rispetto al reverse mode. **Si calcolano le derivate parziali di *tutti gli output* rispetto a *un input* in un passaggio**.

- **a.09** Describe the differences between forward mode differentiation and reverse mode differentiation in terms of computational efficiency and application.<br>
    **Response**: Il reverse mode è l'ideale quando la dimensione dell'input è molto maggiore di quella dell'output (quindi come capita quasi sempre nel machine learning). Inoltre su ogni ramo abbiamo un solo derivative value.


## b. Tensor algebra and PyTorch

- **b.01** Explain the concept of a tensor in PyTorch. <br>
**Response** Un tensor in PyTorch è un array multi-dimensionale di numeri che generalizza scalari, vettori e matrice.
  ![Tensore](./images/tensor.png)
- **b.02** How do the addition and multiplication between tensors work? <br>
   **Response* ![Tensor Algebra](./images/tensorAlgebra.png) <br>
   In PyTorch è possibile fare operazioni tra tensori di dimensioni diverse grazie al broadcasting.
  ![Broadcasting](./images/broadcasting.png)
- **b.03** What is the difference between the 'reshape' and 'view' methods in tensor manipulation? <br>
  **Response** Entrambe servono per manipolare le dimensione del tensore, ma 'reshape' modifica la dimensione del tensore in memoria (rearrange più lento, ma accesso più veloce), mentre 'view' non cambia la shape in memoria, ma solo l'indexing (acesso più lento ma rearragne più veloce). In pytroch, il rearrange segue il row-major order.
    ![Tensore rearrange](./images/tensorAlgebra.png)
- **b.05**/**b.06** Describe the process of creating a custom dataset using 'torch.utils.data.Dataset'. What is the 'torch.utils.data.IterableDataset' and how is it used? <br>
**Response** Per creare un custom Dataset bisogna:
  - creare una classe derivata da torch.utils.data.Dataset
  - implementare i metodi `__init__`,`__len__` e `__getitem__` per creare un dataset che segue il map-model
  - per creare un dataset iterable invece bisogna implementare i metodi `__iter__`, `__next__` che restituiscono rispettivamente `self` e il prossimo data point 
- **b.07** How does the 'torch.utils.data.DataLoader' work in PyTorch? <br>
**Response**: Il Dataloader combina il dataset e il sampler e fornisce un iterable sul dataset (supporta entrambi i tipi di dataset). Utile perchè ha la funzione di automatica batching. Generalmente usato come `torch.utils.data.DataLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True)`
- **b.08**/**b.09** Describe the structure and purpose of the 'torch.nn.Module' class. What are the key methods in a PyTorch module, and how are they implemented? <br>
**Response**: Indubbiamente il modulo più importante e alla base della costruzione di reti neurali. Ogni modello ha dei parametri che sono wrappati in un tensore `torch.nn.Parameter`. Alcuni dei moduli già forniti sono `torch.nn.Linear, torch.nn.ReLU. torch.nn.Sequential`. <br> Per creare un modulo custom:
  - creare una classe che eredita da `torch.nn.Module`
  - implementare il metodo `__init__`
  - implementare il metodo `forward(self,input)`. In questo metodo avviene la computazione (il passo forward della rete neurale).
Tra i metodo presenti ricordiamo `parameters` (restituisce un iteratore sui parametri) e `to` (sposta il modulo da device ad un altro, ad esempio da CPU a GPU).
- **b.11** Explain the concept and application of batch size in model training. <br>
**Response**: Meglio utilizzare un sottoinsieme del dataset piuttosto che calcolare la Loss Function (e il gradiente) su tutto il Dataset che può essere molto grande.
- **b.12** How do you implement a simple linear regression model in PyTorch? <br>
   **Response**: Dopo aver creato un Modulo con un unico parametro w, e aver instanziato l'oggetto:
   ```optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
      dl = torch.utils.data.DataLoader(dataset = ds,batch_size=8)
      loss_fn = torch.nn.MSELoss()
      for epoch in range(0, epochs):
         for batch in dataloader:
            y_model = model(batch.input)
            error = loss_fn(y_model, batch.target)
            optimizer.zero_grad()      # per azzerare i gradienti
            error.backward()
            optimizer.step()
   ```

## c. Convolutional Neural Networks and ResNets

- **c.01** What is a CNN?
- **c.02** How does the convolution operation work in CNNs?
- **c.03** Explain the significance of kernel size, padding, and stride in convolutional layers.
- **c.04** What are the roles of pooling layers in CNNs?
- **c.05** Discuss the function of activation functions in CNNs.
- **c.06** How does a Conv2D layer in PyTorch operate?
- **c.07** Explain the concept of channels in CNNs and their significance.
- **c.08** What are skip connections in CNNs, and how do they function?
- **c.09** Define a Residual Network (ResNet) and its advantages in deep learning.
- **c.10** Draw a diagram of a ResNet and its computational graph
- **c.11** Explain the concept of feature maps in CNNs
- **c.12** Describe the architecture of a typical CNN.
- **c.13** What are the common challenges in training deep CNNs?
- **c.14** How do residual blocks in ResNets mitigate the vanishing gradient problem?
- **c.15** Discuss the application of CNNs in image classification tasks, with an example like MNIST.
- **c.16** How does the structure of a ResNet differ from a standard CNN?

## d. Recurrent Neural Networks

- **d.01** What is a Recurrent Neural Network (RNN), and how does it work?
- **d.02** Explain the concept of time-varying inputs and outputs in RNNs.
- **d.03** Describe two major application families of RNNs: Sequence to Task and Sequence to Sequence.
- **d.04** How do RNNs capture temporal dependencies and patterns in sequences?
- **d.05** Discuss the vanishing gradient problem in RNNs and its impact on learning from long sequences.
- **d.06**  Draw a diagram of a RNN and its computational graph
- **d.07** What are Long Short-Term Memory (LSTM) networks and how do they address the vanishing gradient problem?
- **d.08** What are Gated Recurrent Unit (GRU) networks and how do they differ from LSTMs?
- **d.09** Describe the functionality of the three gates in GRUs: reset, update, and new gate.
- **d.10** Explain how to train an RNN for a sequence classification problem.
- **d.11** Discuss the considerations for setting up RNNs, LSTMs, and GRUs for a classification task
- **d.12** Discuss the challenges in training RNNs and how they can be mitigated.

## e. Autoencoders and VAEs

- **e.01** What is an autoencoder, and what are its primary components?
- **e.02** Describe the roles of the encoder and decoder in an autoencoder.
- **e.03** What is meant by the "latent space" in an autoencoder?
- **e.04** How does an autoencoder learn a compact representation of input data?
- **e.05** Discuss the concept of reconstruction error in autoencoders.
- **e.06** What is a denoising autoencoder, and how does it differ from a traditional autoencoder?
- **e.07** What are Variational Autoencoders, and how do they differ from regular autoencoders?
- **e.08** How does the Reparametrization Tick work?
- **e.09** Draw a diagram of a VAE and its computational graph
- **e.10** Why VAEs are a 'generative' architecture?
- **e.11** What is the Kullback-Leibler divergence, and how is it used in VAEs?
- **e.12** Describe a practical implementation of training a simple autoencoder on the MNIST dataset.
- **e.13** Explain the procedure for training a Variational Autoencoder on the MNIST dataset.
- **e.14** How does the 'Face Swap' algorithm work?

## f. Vector Quantized Variational Autoencoders

- **f.01** What is a Vector Quantized Variational Autoencoder (VQ-VAE)?
- **f.02** Explain the process of mapping input data to a continuous latent space and then to discrete codes in a VQ-VAE.
- **f.03** How does the vector quantization process work in a VQ-VAE, and what is the role of the codebook?
- **f.04** How does the quantization trick work?
- **f.05** Draw a diagram of a VQ-VAE and its computational graph
- **f.06** Describe the function of the 'cdist' function in PyTorch in the context of VQ-VAEs.
- **f.07** What is the responsibility of the decoder in a VQ-VAE, and how does it utilize discrete codes for data reconstruction or generation?
- **f.08** Discuss the types of losses used in training a VQ-VAE, specifically reconstruction, codebook, and commitment loss.
- **f.09** Explain the practical steps involved in training a VQ-VAE on the MNIST dataset.
- **f.10** Explain how to generate images from random codes in a VQ-VAE and the insights it provides.

## g. Generative Adversarial Networks

- **g.01** What are Generative Adversarial Networks?
- **g.02** Describe the architecture of GANs, including the roles of the generator and discriminator.
- **g.03** Explain how the discriminator functions as a binary classifier in GANs.
- **g.04** Outline the steps involved in training the discriminator in a GAN.
- **g.05** Explain the training process of the generator in a GAN.
- **g.06** Draw a diagram of a GAN and its computational graph
- **g.07** What is an Auxiliary Classifier GAN (ACGAN), and how does it differ from standard GANs?
- **g.08** Discuss the challenges in training GANs and strategies to overcome them.
- **g.09** Describe the process of training a Deep Convolutional GAN (DCGAN) on the MNIST dataset.
- **g.10** How can the balance between randomness and class information be maintained in an ACGAN?

## h. Advanced Architectures

- **h.01** What is YOLO (You Only Look Once) in the context of object detection, and how does it perform real-time detection?
- **h.02** Explain how YOLO divides an image into a grid for object detection and how it predicts bounding boxes and class probabilities.
- **h.03** Describe Non-Maximum Suppression (NMS) and its role in object detection.
- **h.04** What is Faster R-CNN, and how does it combine a Region Proposal Network (RPN) with a CNN?
- **h.05** Explain the U-Net architecture and its application in semantic image segmentation.
- **h.06** What is CLIP (Contrastive Language-Image Pretraining) by OpenAI, and how does it combine vision and language?
- **h.07** Discuss Denoising Diffusion Probabilistic Models (DDPM) and Latent Diffusion, and their role in generating high-quality samples.
- **h.08** Describe the DALL-E 2 architecture
