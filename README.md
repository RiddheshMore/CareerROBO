

# Personalized Career Coach - Socially Interactive Robot

## Project Overview

This project aims to build a socially interactive robot that acts as a career coach, helping students find suitable career options based on their preferences. The robot, named **CareerRobo**, will use the Pepper robot in the qiBullet simulation environment and offer various features, including face detection, multi-modal greetings, personalized career suggestions, and interactive conversations. Additionally, the system will integrate a Bayesian network to infer and recommend career options based on user preferences.

## Key Features

- **Face Detection & Verification**: The robot detects faces from a video stream and optionally verifies authorized users.
- **Multi-modal Greeting**: After detecting a face, the robot greets the user with personalized multi-modal behavior.
- **Career Suggestion**: The robot helps users find suitable career options based on their preferences using an internal model.
- **Gestures & Speech**: Illustrative gestures accompany speech to enhance communication.
- **Abusive Language Filtering**: The robot detects and avoids engaging in abusive conversations.
- **Farewell**: The robot bids farewell in a socially appropriate manner after the conversation.
- **Optional Speech Input**: The robot can accept and process speech input.
  
## Agent Name

**CareerRobo**

## Slogan

**"Empowering Your Future, One Career at a Time!"**

## Project Timeline

- **28 Nov 2024**: Project Start
- **12 Dec 2024**: Checkpoint 1 - Proposal submission (design of software architecture & Bayesian network)
- **09 Jan 2025**: Checkpoint 2 - Upload video (face detection, greeting, dialog flow, gestures, farewell, Bayesian model)
- **23 Jan 2025**: Final project demonstration (working prototype, 5-slide presentation, Q&A)

## Requirements

- **Software**:
  - Pepper Robot simulation in qiBullet environment
  - RASA for dialogue management
  - Python for backend programming
  - Bayesian network library for career preference inference
  
- **Hardware**:
  - Webcam for face detection
  - Pepper Robot (for simulation or real hardware if available)




## Running Tests



To run tests, run the following commands

```bash
  rasa run actions
```
In a seperate terminal run,
```bash
  rasa run --enable-api --cors "*"
```
Open another terminal and run,
```bash
  career_robo.py
```