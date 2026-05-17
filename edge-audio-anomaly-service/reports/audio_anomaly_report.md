# Edge Audio Anomaly Report

- Clips analyzed: `13`
- Anomaly events: `5`
- Figure: `E:\linux\edge-audio-anomaly-service\reports\audio_score_curve.png`

![Audio score curve](audio_score_curve.png)

| File | Label | Score | Threshold | Anomaly |
|---|---|---:|---:|---|
| normal_00.wav | normal | 2.118 | 5.065 | False |
| normal_01.wav | normal | 2.678 | 5.065 | False |
| normal_02.wav | normal | 1.142 | 5.065 | False |
| normal_03.wav | normal | 2.568 | 5.065 | False |
| normal_04.wav | normal | 2.312 | 5.065 | False |
| normal_05.wav | normal | 3.771 | 5.065 | False |
| normal_06.wav | normal | 2.279 | 5.065 | False |
| normal_07.wav | normal | 1.518 | 5.065 | False |
| anomaly_00.wav | anomaly | 12401.409 | 5.065 | True |
| anomaly_01.wav | anomaly | 12537.771 | 5.065 | True |
| anomaly_02.wav | anomaly | 12358.132 | 5.065 | True |
| anomaly_03.wav | anomaly | 12507.392 | 5.065 | True |
| anomaly_04.wav | anomaly | 12414.175 | 5.065 | True |

## Linux Application Notes

WAV replay stands in for microphone or UDP audio input. The feature and
model boundary can stay stable when the input source becomes ALSA, PulseAudio,
or an embedded recorder process.
