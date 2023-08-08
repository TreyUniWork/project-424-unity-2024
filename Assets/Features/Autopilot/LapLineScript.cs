using UnityEngine;
using Perrinn424.AutopilotSystem;

public class LapLineScript : MonoBehaviour
{
    public Autopilot autopilot;
    private bool isFirstTime = true;
    private int collisionCounter = 0;  // New counter for colliders

    void OnTriggerEnter(Collider other)
    {

            collisionCounter++;
            
            // Only process every 4th collision, assuming there are 4 colliders on the vehicle
            if (collisionCounter % 4 == 0)
            {
                if (isFirstTime)
                {
                    isFirstTime = false;
                }
                else if (!autopilot.HasCompletedLap)
                {
                    autopilot.CompleteLap();
                    Debug.Log("Lap completed!");
                }
            }
        }
}
