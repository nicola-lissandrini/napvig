# napvig
ROS-ready navigation algorithm

`roslaunch napvig napvig.launch`

## Policy-driven trajectory generation
The new version of Napvig introduces the possiblity to specify a Policy according to which the trajectory is generated.
The generation is handled by `NapvigPrediction`, according to the following pseudocode:

`followPolicy`:
* `policy.init` 
* do
* * `policy.getFirstSearch` first search sample
* * `predictTrajectory (policy)`
* until `selectTrajectory` choose the computed trajectory

`predictTrajectory`:
* do
* * `policy.getNextSearch` next search direction
* * `napvig.step`
* until `policy.terminationCondition = PREDICTION_TERMINATION_NONE`

Then you will need to inherit the abstract class `NapvigPrediction` and implement
`trajectoryAlgorithm`, calling `followPolicy` with your desired policy,
according to your specific procedure that might involve policy switching.
