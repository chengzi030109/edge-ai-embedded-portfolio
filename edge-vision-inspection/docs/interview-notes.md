# Interview Notes

This project is an embedded Linux vision inspection app. It uses folder replay
as the hardware-free stand-in for USB camera or RTSP input, extracts lightweight
image statistics, runs anomaly scoring, and saves annotated outputs.

Key talking points:

- Folder replay lets the app be tested without a camera.
- The input/model/report boundary can later host ONNX object detection.
- Annotated outputs and score reports make demo behavior visible.
- systemd service files show Linux deployment shape.

