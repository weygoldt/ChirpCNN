# DeepChirps 

Let's try to improve chirp detection with deep learning.

## Approach

1. Simulate a labeled dataset 
2. Train a binary classifier to detect chirps in spectrogram snippets
3. Slide a window along the frequency track of a single track and apply the classifier on each sliding window
4. Clean-up: Group chirps if they are detected multiple times

## Issues

- A chirp only lasts for 20-200 ms but the anomaly it introduces on a spectrogram with sufficient frequency resolution lasts up to a second. 
- The classifier might be able to detect chirps well, but assigning them to the correct emitter is a seperate problem.

## To do 

- [x] Create a synthetic dataset 
- [ ] Add more variation to the dataset
- [ ] Build a classifier
- [ ] Implement window-sliding on actual data 
- [ ] Output validation on real data
