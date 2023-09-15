using UnityEngine;
using UnityEngine.SceneManagement;

public class ZeroSpeedDetector : MonoBehaviour
{
    public float speedThreshold = 0.5f; // Speed below which the vehicle is considered stopped

    private Rigidbody vehicleRb;
    private VehiclePhysics.Timing.LapTimer lapTimer;
    private bool hasStarted = false;
    private float delayTime = 1.0f; // Delay time in seconds

    private void Start()
    {
        vehicleRb = GetComponent<Rigidbody>();

        // Find an instance of the LapTimer in the scene
        lapTimer = FindObjectOfType<VehiclePhysics.Timing.LapTimer>();

        if (lapTimer == null)
        {
            Debug.LogError("No LapTimer found in the scene.");
        }
    }

    private void Update()
    {
        // Check if the delay time has passed and the action hasn't started yet
        if (!hasStarted && Time.time >= delayTime)
        {
            hasStarted = true; // Set the flag to indicate that the action has started
            CheckSpeedAndReset();
        }
    }

    private void CheckSpeedAndReset()
    {
        float currentSpeed = vehicleRb.velocity.magnitude;

        if (currentSpeed <= speedThreshold)
        {
            // Vehicle is considered to have stopped
            Debug.Log("Vehicle stopped!");

            // Invalidate the lap
            if (lapTimer != null)
            {
                lapTimer.InvalidateLapDueToStop();
            }

            // Reset the current scene
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
    }
}
