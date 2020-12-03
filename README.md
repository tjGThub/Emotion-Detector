# Emotion Detector
This program takes a jpg image as an input and determines the emotion of the person inside the image. It is currently able to detect only anger, happy and neutral emotions. This model is trained using only anger, happy and neutral data of [FER-2013 dataset](https://www.kaggle.com/msambare/fer2013) from Kaggle. Further improvements will be done to improve the models capability. 

# Motivation
To improve interaction between human and machine.

# Network Architecture

# Steps
1. Download the whole folder.
2. Download the FER-2013 dataset.
3. Import pom.xml as project in IntelliJ.
4. Once the dependencies are resolved, open the test1.java file.
5. Copy and paste angry, neutral and happy files for test and train into the resource file and rename the file path for training purposes. (rename the modelFilename to create a new model as the default model has been created.)
6. Change image path to own image path under testImage() function.
7. Run the program.
8. The output stating the emotion of the person will be shown in the console.

# Future Development
1. Ability to detect more emotions.
2. Ability to perform localization using bounding boxes.
3. Apply transfer learning.
4. Ability to detect emotions using webcam video as input.
5. Solve the problem of overfitting.

# Contact Information
1. Tang Jie                             Email: tjjust4work@gmail.com
2. Muhammad Khairul Asyraf bin Suaimi   Email: mkhairulasyraf12@gmail.com
3. Muhamad Noorazizi Bin Abd Ghani      Email: azizi.ghani@gmail.com
