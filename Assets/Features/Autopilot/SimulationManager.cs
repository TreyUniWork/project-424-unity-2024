using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Perrinn424.AutopilotSystem;

public class SimulationManager : MonoBehaviour
{

public Autopilot autopilot;  // Reference to the Autopilot script
    private int lapCount = 0;
    private string[] assetPaths = {"GeneticAssets/asset1", "GeneticAssets/asset2"};  // Add paths to your asset files
    private int currentAssetIndex = 0;

    void Start()
    {
                // Start the simulation by enabling the Autopilot script
        autopilot.enabled = true;

        // Load the first asset file
        RecordedLap firstLap = Resources.Load<RecordedLap>(assetPaths[0]);
        autopilot.recordedLap = firstLap;

        if (firstLap != null)
        {
            Debug.Log("Loaded initial asset: " + assetPaths[0]);
        }
        else
        {
            Debug.LogError("Failed to load initial asset: " + assetPaths[0]);
        }
    
    }

    void Update()
    {
        if (autopilot.HasCompletedLap)
        {
            lapCount++;
            Debug.Log("Lap completed. Total laps: " + lapCount);

            if (lapCount == 1)
            {
                // Load the next asset file
                currentAssetIndex = (currentAssetIndex + 1) % assetPaths.Length;
                RecordedLap newLap = Resources.Load<RecordedLap>(assetPaths[currentAssetIndex]);
                autopilot.recordedLap = newLap;

                if (newLap != null)
                {
                    Debug.Log("Loaded asset: " + assetPaths[currentAssetIndex]);
                }
                else
                {
                    Debug.LogError("Failed to load asset: " + assetPaths[currentAssetIndex]);
                }

                // Reset lap count
                lapCount = 0;
                Debug.Log("Starting a new lap with asset: " + assetPaths[currentAssetIndex]);

            }

            // Start a new lap
            autopilot.StartNewLap();
        }
    }
    IEnumerator LoadNewLap(string assetPath)
    {
        RecordedLap newLap = Resources.Load<RecordedLap>(assetPath);
        autopilot.recordedLap = newLap;

        if (newLap != null)
        {
            Debug.Log("Loaded asset: " + assetPath);
        }
        else
        {
            Debug.LogError("Failed to load asset: " + assetPath);
        }

        // Wait until the end of the frame to start the new lap
        yield return new WaitForEndOfFrame();

        // Start a new lap
        autopilot.StartNewLap();
    }

    private bool AutopilotHasCompletedALap()
    {
        return autopilot.HasCompletedLap;
    }
}
