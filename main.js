//Imports
import {FilesetResolver, PoseLandmarker, DrawingUtils} from '@mediapipe/tasks-vision';
import poseData3 from './data/poseData3.json' assert {type: 'json'}
import poseData4 from './data/poseData4.json' assert {type: 'json'}

//Wait for the window to load before doing anything
window.addEventListener('load', init);

//Variables
let videoElement;
let canvas;
let poseSelectButtonContainer;
let webcamControlContainer;
let webcamContainer;

let poseLandmarker;
let isWebcamRunning = false;
let canvasContext;
let drawingUtils;

let startTimer;
let stopTimer;
let interval;
let frames = [];
let currentPose;

let currentData = [];

const nnOptions = {
    task: 'classification',
    debug: true,
    layers: [
        {
            type: 'dense',
            units: 32,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 8,
            activation: 'relu',
        },
        {
            type: 'dense',
            units: 16,
            activation: 'relu',
        },
        {
            type: 'dense',
            activation: 'softmax',
        },
    ],
    learningRate: 0.15
}

ml5.setBackend("webgl");
const nn = ml5.neuralNetwork(nnOptions);


//Functions
async function init() {

    //Get all the DOM elements we need
    videoElement = document.getElementById('webcamElement');
    canvas = document.getElementById('landmarkOverlay');
    poseSelectButtonContainer = document.getElementById('poseSelectContainer');
    webcamControlContainer = document.getElementById('webcamControls');
    webcamContainer = document.getElementById('webcamContainer');

    //Get the canvas context
    canvasContext = canvas.getContext('2d');

    drawingUtils = new DrawingUtils(canvasContext);

    //Clear out the canvas just in case
    clearCanvas();

    //Load json data into nn
    trainNn();

    try {

        //Wait for the pose landmarker to be created before allowing the buttons to work
        await createPoseLandmarker();

        //Add event listeners to buttons
        poseSelectButtonContainer.addEventListener('click', selectPose);
        webcamControlContainer.addEventListener('click', controlWebcam);

    } catch (error) {
        console.error(error.message);
    }

}

async function createPoseLandmarker() {
    const vision = await FilesetResolver.forVisionTasks('./node_modules/@mediapipe/tasks-vision/wasm');

    // poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    //     baseOptions: {
    //         modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    //         delegate: 'GPU'
    //     },
    //     runningMode: 'IMAGE',
    //     numPoses: 1
    // });

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "./pose_landmarker_lite.task",
            delegate: 'GPU'
        },
        runningMode: 'IMAGE',
        numPoses: 1
    });
}

function selectPose(e) {

    //Don't do anything if the webcam isn't running or if the pressed thing wasn't a button
    if (!isWebcamRunning || e.target.tagName !== 'BUTTON') {
        return;
    }

    currentPose = e.target.id;

    //Reset the timers just in case
    clearInterval(interval);
    clearTimeout(stopTimer);
    clearTimeout(startTimer);

    //Make sure to clear any previous frames too if there were any
    frames = [];

    console.log('Prepare to pose');

    //The timers here wait 3 seconds before collecting frames
    //The collection happens ever 50 ms
    //The collecting stops after 5 seconds
    startTimer = setTimeout(() => {

        console.log('Pose collection in progress');

        interval = setInterval(collectFrame, 50);
        stopTimer = setTimeout(storePoses, 4000);

    }, 3000);


}

async function controlWebcam(e) {

    //Don't do anything if the pressed thing wasn't a button
    if (e.target.tagName !== 'BUTTON') {
        return;
    }

    if (e.target.id === 'startWebcam') {

        if (isWebcamRunning) {
            videoElement.play();
            clearCanvas();
            return;
        }

        await startWebcam();
        return;
    }

    if (e.target.id === 'stopWebcam') {

        videoElement.srcObject.getTracks()[0].stop();

        isWebcamRunning = false;

        return;
    }

    if (e.target.id === 'testDetection') {

        clearTimeout(startTimer);

        startTimer = setTimeout(testPose, 2000);

        return;

    }

    if (e.target.id === 'saveData') {

        const json = JSON.stringify(currentData);

        let blob = new Blob([json], {type: 'application/json'});
        let url = URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.href = url;
        a.download = 'trainingData.json';
        a.click();
        URL.revokeObjectURL(url);

    }

}

async function startWebcam() {

    try {

        videoElement.srcObject = await navigator.mediaDevices.getUserMedia({video: true, audio: false});

        videoElement.addEventListener("loadeddata", () => {

            canvas.style.width = videoElement.videoWidth;
            canvas.style.height = videoElement.videoHeight;

            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;

            webcamContainer.style.height = videoElement.videoHeight + "px";

            isWebcamRunning = true;
        });

    } catch (error) {
        console.error(error.message);
    }

}

function detectPose() {

    const result = poseLandmarker.detect(videoElement);

    return result.landmarks[0];

}

function collectFrame() {

    frames.push(new VideoFrame(videoElement));

}

function storePoses() {

    //Stop the collection of poses
    clearInterval(interval);

    const simplifiedData = [];

    for (const frame of frames) {
        let newPose = detectPose(frame);

        if (!newPose) {
            continue;
        }

        const simplifiedPose = {
            data: [],
            label: currentPose
        };

        for (const landmark of newPose) {
            simplifiedPose.data.push(landmark.x);
            simplifiedPose.data.push(landmark.y);
            simplifiedPose.data.push(landmark.z);
        }

        simplifiedData.push(simplifiedPose);

    }

    currentData = currentData.concat(simplifiedData);
    console.log(`done recording for ${currentPose}`);

}

function clearCanvas() {
    canvasContext.clearRect(0, 0, canvas.width, canvas.height);
}

function drawPose(pose) {
    drawingUtils.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, {color: "#FFDDEE", lineWidth: 5});
    drawingUtils.drawLandmarks(pose, {radius: 4, color: "#FF00FF", lineWidth: 2});
}

async function trainNn() {

    let poses = poseData3;

    poses = poses.concat(poseData4);

    let currentIndex = poses.length - 1;

    while (currentIndex >= 0) {

        const randomIndex = Math.floor(Math.random() * poses.length);

        const poseA = poses[currentIndex];
        const poseB = poses[randomIndex];

        poses[randomIndex] = poseA;
        poses[currentIndex] = poseB;

        currentIndex--;

    }

    const trainingData = poses.slice(0, Math.floor(poses.length * 0.8))
    const testingData = poses.slice(Math.floor(poses.length * 0.8) + 1)

    for (const pose of trainingData) {
        nn.addData(pose.data, {label: pose.label});
    }

    console.log('NN filled with data!');

    nn.normalizeData();
    await nn.train({epochs: 50}, () => console.log('Finished Training!'));

    let correctAnswers = 0;
    let correctHandsUp = 0;
    let correctEyesCovered = 0;
    let correctFakeSurprise = 0;

    for (const pose of testingData) {

        const answer = await nn.classify(pose.data);

        if (answer[0].label === pose.label) {
            correctAnswers++;

            switch(pose.label) {
                case('handsUp'): correctHandsUp++; break;
                case('eyesCovered'): correctEyesCovered++; break;
                case('fakeSurprise'): correctFakeSurprise++; break;
            }

        } else {
            console.log(`NN thinks this is a ${answer[0].label} while the actual label is ${pose.label}`);
        }


    }

    const accuracy = correctAnswers / testingData.length;

    console.log(`Got ${correctAnswers} correct answers out of ${testingData.length}`);
    console.log(`hands up: ${correctHandsUp}, eyes covered: ${correctEyesCovered}, fake surprise: ${correctFakeSurprise}`);
    console.log(`Accuracy of the model is about ${accuracy * 100}%`);

    // nn.save("poseDetection", () => console.log("model was saved!"));

}

async function testPose() {
    videoElement.pause();

    canvasContext.drawImage(videoElement, 0, 0);

    const pose = detectPose();
    drawPose(pose);
    console.log(pose);

    isWebcamRunning = false;
    videoElement.srcObject.getTracks()[0].stop();

    const simplifiedPose = [];

    for (const landmark of pose) {
        simplifiedPose.push(landmark.x);
        simplifiedPose.push(landmark.y);
        simplifiedPose.push(landmark.z);
    }

    const result = await nn.classify(simplifiedPose);

    console.log(result);
}