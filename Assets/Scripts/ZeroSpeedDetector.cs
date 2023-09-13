using UnityEngine;
using UnityEngine.SceneManagement;

public class ZeroSpeedDetector : MonoBehaviour
{
    public float speedThreshold = 0.5f; // Speed below which the vehicle is considered stopped

    private Rigidbody vehicleRb;
    private VehiclePhysics.Timing.LapTimer lapTimer;

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
