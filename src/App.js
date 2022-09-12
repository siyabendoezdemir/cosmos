import "./App.css";
// eslint-disable-next-line
import * as tfJsCore from "@tensorflow/tfjs-core";
// eslint-disable-next-line
import * as tfJsConverter from "@tensorflow/tfjs-converter";
// eslint-disable-next-line
import * as tfJsBackendWebgl from "@tensorflow/tfjs-backend-webgl";

import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
// import {isMobile} from 'react-device-detect';
import * as poseDetection from "@tensorflow-models/pose-detection";

const VIDEO_CONSTRAINTS = {
  facingMode: "environment",
  deviceId: "",
  frameRate: { max: 8, ideal: 8 },
  // width: isMobile ? 360 : 1280,
  // height: isMobile ? 270 : 720
};
const MOVENET_CONFIG = {
  maxPoses: 1,
  type: "lightning",
  scoreThreshold: 0.3,
};

const DEFAULT_LINE_WIDTH = 2;
const DEFAULT_RADIUS = 4;

let detector;
let startInferenceTime,
  numInferences = 0;
let inferenceTimeSum = 0,
  lastPanelUpdate = 0;
let rafId;
let canvasFullScreen;
let ctxFullScreen;
let model;
let modelType;
let mousePushed = false;
let initialHeadX = [0, 0];
let initialHeadY = [0, 0];

var ballRadius = 10;

var vx = 1,
  vy = 2.5,
  gravity = 0.001,
  damping = 0.001,
  traction = 0.001,
  paused = false;

let ballX;
let ballY;

function App() {
  const [cameraReady, setCameraReady] = useState(false);
  const [displayFps, setDisplayFps] = useState(0);
  const [leftHandCoords, setLeftHandCoords] = useState(0);
  const [rightHandCoords, setRightHandCoords] = useState(0);
  const [fell, setFall] = useState(false);

  const webcamRef = useRef({});

  useEffect(() => {
    _loadPoseNet().then();

    // eslint-disable-next-line
  }, []);

  const _loadPoseNet = async () => {
    if (rafId) {
      window.cancelAnimationFrame(rafId);
      detector.dispose();
    }

    detector = await createDetector();
    await renderPrediction();
  };

  const createDetector = async () => {
    model = poseDetection.SupportedModels.MoveNet;
    modelType = poseDetection.movenet.modelType.SINGLEPOSE_THUNDER; //or SINGLEPOSE_THUNDER
    return await poseDetection.createDetector(model, { modelType: modelType });
  };

  const renderPrediction = async () => {
    await renderResult();
    rafId = requestAnimationFrame(renderPrediction);
  };

  const renderResult = async () => {
    const video = webcamRef.current && webcamRef.current["video"];

    if (!cameraReady && !video) {
      return;
    }

    if (video.readyState < 2) {
      return;
    }

    beginEstimatePosesStats();
    const poses = await detector.estimatePoses(video, {
      maxPoses: MOVENET_CONFIG.maxPoses, //When maxPoses = 1, a single pose is detected
      flipHorizontal: false,
    });
    endEstimatePosesStats();
    drawCtxFullScreen(video);

    if (poses.length > 0) {
      drawResultsFullScreen(poses);
    }
  };

  const beginEstimatePosesStats = () => {
    startInferenceTime = (performance || Date).now();
  };

  const endEstimatePosesStats = () => {
    const endInferenceTime = (performance || Date).now();
    inferenceTimeSum += endInferenceTime - startInferenceTime;
    ++numInferences;
    const panelUpdateMilliseconds = 1000;

    if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
      const averageInferenceTime = inferenceTimeSum / numInferences;
      inferenceTimeSum = 0;
      numInferences = 0;
      setDisplayFps(1000.0 / averageInferenceTime, 120);
      lastPanelUpdate = endInferenceTime;
    }
  };

  const drawCtxFullScreen = (video) => {
    canvasFullScreen = document.getElementById("output-full-screen");
    ctxFullScreen = canvasFullScreen.getContext("2d");

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    video.width = videoWidth;
    video.height = videoHeight;

    canvasFullScreen.width = videoWidth;
    canvasFullScreen.height = videoHeight;
    ctxFullScreen.fillRect(0, 0, videoWidth, videoHeight);

    //ctxFullScreen.translate(video.videoWidth, 0);
    //ctxFullScreen.scale(-1, 1);

    //Disable if you want to disable the video
    ctxFullScreen.drawImage(video, 0, 0, videoWidth, videoHeight);
  };

  const drawResultsFullScreen = (poses) => {
    for (const pose of poses) {
      drawResult(pose);
    }
  };

  const drawResult = (pose) => {
    if (pose.keypoints != null) {
      drawKeypoints(pose.keypoints);
      drawSkeleton(pose.keypoints);
    }
  };

  document.body.onmousedown = function () {
    mousePushed = true;
  };
  document.body.onmouseup = function () {
    mousePushed = false;
  };
  document.body.oncontextmenu = function (e) {
    e.preventDefault();
  };

  function checkFall(nose) {
    initialHeadY.push(nose.y);
    initialHeadX.push(nose.x);

    if (
      nose.y - initialHeadY[initialHeadY.length - 2] > 20 &&
      initialHeadY[initialHeadY.length - 2] != 0
    ) {
      setTimeout(
        function () {
          //Start the timer

          var checkActualFall = setInterval(
            function () {
              console.log("checkMove");
              if (
                nose.y - initialHeadY[initialHeadY.length - 2] > 30 ||
                initialHeadY[initialHeadY.length - 2] - nose.y > 30 ||
                nose.x - initialHeadX[initialHeadX.length - 2] > 40 ||
                initialHeadX[initialHeadX.length - 2] - nose.x > 40
              ) {
                setFall(false);
                clearInterval(checkActualFall);
              } else {
                setFall(true);
              }
            }.bind(this),
            100
          );
        }.bind(this),
        100
      );
    }
  }

  const drawKeypoints = (keypoints) => {
    const keypointInd = poseDetection.util.getKeypointIndexBySide(model);
    ctxFullScreen.fillStyle = "White";
    ctxFullScreen.strokeStyle = "White";
    ctxFullScreen.lineWidth = DEFAULT_LINE_WIDTH;

    checkFall(keypoints[0]);

    setLeftHandCoords(
      `X: ${keypoints[9].x.toFixed(1)} \n Y: ${keypoints[9].y.toFixed(1)}`
    );
    setRightHandCoords(
      `X: ${keypoints[10].x.toFixed(1)} \n Y: ${keypoints[10].y.toFixed(1)}`
    );

    for (const i of keypointInd.middle) {
      drawKeypoint(keypoints[i]);
    }

    ctxFullScreen.fillStyle = "Green";

    for (const i of keypointInd.left) {
      drawKeypoint(keypoints[i]);
    }

    ctxFullScreen.fillStyle = "Orange";

    for (const i of keypointInd.right) {
      drawKeypoint(keypoints[i]);
    }
  };

  const drawKeypoint = (keypoint) => {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = MOVENET_CONFIG.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      ctxFullScreen.beginPath();

      ctxFullScreen.arc(keypoint.x, keypoint.y, DEFAULT_RADIUS, 0, 2 * Math.PI);
      ctxFullScreen.fill();
      ctxFullScreen.stroke();
    }
  };

  const drawSkeleton = (keypoints) => {
    ctxFullScreen.fillStyle = "White";
    ctxFullScreen.strokeStyle = "White";
    ctxFullScreen.lineWidth = DEFAULT_LINE_WIDTH;
    poseDetection.util.getAdjacentPairs(model).forEach(([i, j]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j]; // If score is null, just show the keypoint.

      const score1 = kp1.score != null ? kp1.score : 1;
      const score2 = kp2.score != null ? kp2.score : 1;
      const scoreThreshold = MOVENET_CONFIG.scoreThreshold || 0;

      if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
        ctxFullScreen.beginPath();
        ctxFullScreen.moveTo(kp1.x, kp1.y);
        ctxFullScreen.lineTo(kp2.x, kp2.y);
        ctxFullScreen.stroke();
      }
    });
  };

  const onUserMediaError = () => {
    console.log("ERROR in Camera!");
  };

  const onUserMedia = () => {
    console.log("Camera loaded!");
    setCameraReady(true);
  };

  /*

  make a function that constantly checks the users state (Standing, laying) and logs it to the console

  To achieve this you can check for the Y value of the user. His/Her Y-Value will always be low if he/she is laying down

  Check for the Y axis and if it's below a certain point then set the status to laying

  */

  return (
    <section className="App">
      <div className="recording_container">
        <Webcam
          className="webcam"
          style={{ visibility: "hidden" }}
          ref={webcamRef}
          audio={false}
          // height={isMobile ? 270 : 720}
          // width={isMobile ? 360 : 1280}
          videoConstraints={VIDEO_CONSTRAINTS}
          onUserMediaError={onUserMediaError}
          onUserMedia={onUserMedia}
        />

        <div className="play-area">
          <canvas className="outputCanvas" id="output-full-screen" />
        </div>

        <label className="fps-info">FPS: {displayFps.toFixed()}</label>
        <h2
          className="fallText"
          style={{ visibility: fell ? "visible" : "hidden" }}
        >
          Fall detected
        </h2>
        <label className="hands-info">
          Left Hand: <br /> {leftHandCoords}
        </label>
        <label className="hands-info" id="rightHand">
          Right Hand: <br /> {rightHandCoords}
        </label>
      </div>
    </section>
  );
}

export default App;
