using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Perrinn424.AutopilotSystem;

public class SimulationManager : MonoBehaviour
{
    public Autopilot autopilot;
    private int lapCount = 0;
    private string[] assetPaths = {"GeneticAssets/asset1", "GeneticAssets/asset2"};
    private int currentAssetIndex = 0;

    void Start()
    {
        autopilot.enabled = true;
        LoadLap(assetPaths[currentAssetIndex]);
    }

    void Update()
{
    if (autopilot.HasCompletedLap)
    {
        float finalLapTime = autopilot.GetFinalLapTime();
        Debug.Log("Final lap time: " + finalLapTime);

        lapCount++;
        Debug.Log("Lap completed. Total laps: " + lapCount);

        if (lapCount > 1)
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


    private void LoadLap(string assetPath)
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
    }
}
