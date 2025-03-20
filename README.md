# Simulation and Recover Final Assignment, due 3/20/25

This is an individual, simulate-and-recover project by Alyssa Mendoza, for the EZ diffusion model. ChatGPT was used overall, for structure, naming, organization, and resolving errors.

A common consistency test in cognitive modeling, through trial-and-error and persistent experimentation, I discovered the EZ diffusion model can indeed be used to estimate parameters from generated data. 


Organization-wise, the src folder contains the main Python script, simulate_recover.py, which simulates and recovers parameters using the model. Additionally, the main.sh script runs the complete 3000-iteration simulate-and-recovery exercise. The test folder includes the unit tests in test_simulation.py, which verify stability, as well as parameter sampling and accuracy of recovery, across different sample sizes. The test.sh script runs my test suite. The results folder stores the simulation results in a .csv and also contains two ChatGPT-generated line graphs of the my run simulation’s results. Requirements.txt lists dependencies for running the code.

The results of the exercise, and ChatGPT’s conversion of the csv file into graphs. demonstrate a variety of things:
*Trends in bias: the mean bias for Boundary (α), Drift (ν), and Non-Decision Time (τ) evidently fluctuates at smaller sample sizes. And an increase in sample size correlates with decreasing bias values, reinforcing how larger samples reduce errors in estimation.
*Squared Error Trends: squared error abides by a decreasing trend with the larger sample sizes, so parameter estimates significantly improve the more trials there are. This exemplifies the concept of noise in observed statistics decreasing as N increases.

Ultimately, the results of my project support the EZ Diffusion model’s validity when recovering parameters. A small sample size, like that of N = 10, tends to have higher bias and squared error (unreliable estimates). A larger size, like that of N = 4000, tends to have biases closer to zero and minimal squared errors (true parameters are accurately recovered).

This aligns with Week 9’s lecture slides and shows the EZ diffusion model stands as a reliable model for large data sets, while simultaneously having limitations at smaller sample sizes. The trends of the results additionally align with the predictions theorized in the slides.

My experiment emphasizes how important sample size is when it comes to estimating the parameters of cognitive models. And real-world applications for this include: valuable decision-making studies where researchers need to understand reaction time in perception or decision-making tasks overtime, specifically in psychology or neuroscience, within clinical diagnoses concerning schizophrenia (for altered decision-making), Alzheimer’s (for cognitive decline and longer non-decision times) and ADHD (for lower drift rates).

In terms of potential areas for expansion, I briefly considered adding extra sample sizes like 100, 500, and 10,000 to show whether trends continue as N increases. I had figured this would give me a better idea of if bias and squared error converge toward zero at even larger sample sizes, however I kept encountering errors having to do with the drift rate estimation being unstable for smaller sample sizes and the mean bias for drift rate exceeding the allowed threshold.

