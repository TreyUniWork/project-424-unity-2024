using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Perrinn424.AutopilotSystem;
using VehiclePhysics.Timing;
using System.IO;
using System.Linq;


public class SimulationManager : MonoBehaviour
{
    public Autopilot autopilot;

    /*
    public Autopilot autopilot;
    //private int lapCount = 0;
    public int generations_length = 5; // max number of assets per generation (need to change depending on GA)
    //private string[] assetPaths = { "GeneticAssets/asset1", "GeneticAssets/asset2" };
    private int currentAssetIndex = 0;
    private int currentGeneration = 0;
    [SerializeField] private string generationPrefix = "GEN";
    [SerializeField] private string assetPrefix = "asset";

    void Start()
    {
        //autopilot.enabled = true;

        LoadLap(assetPaths[currentAssetIndex]);
    }

    /*
    void Update()
    {
        if (autopilot.HasCompletedLap)
        {
            //float finalLapTime = autopilot.CalculateDuration();
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

        }
    }
    */

    /*
    public void SwitchOutLap()
    {
        // check what folder to load from
        string basePath = "GeneticAssets/"; // Replace with the actual path to your folders

        string[] folderPaths = Directory.GetDirectories(basePath, "GEN*"); // Get all folders matching the pattern

        if (folderPaths.Length > 0)
        {
            // Sort the folder names in descending order
            var sortedFolders = folderPaths.OrderByDescending(folderPath => GetGenerationNumber(folderPath));

            string latestGenerationFolder = sortedFolders.First(); // Get the latest generation folder

            // Access the desired file within the latest generation folder
            string filePathInLatestGeneration = System.IO.Path.Combine(latestGenerationFolder, "YourFileName.ext");
            Debug.Log("Selected File: " + filePathInLatestGeneration);
        }
        else
        {
            Debug.Log("No generation folders found.");
        }

        // check if which asset file to load

        LoadLap();
    }
    private int GetGenerationNumber(string folderPath)
    {
        string folderName = System.IO.Path.GetFileName(folderPath);
        int generationNumber = 0;

        if (int.TryParse(folderName.Substring(3), out generationNumber))
        {
            return generationNumber;
        }

        return 0; // Return 0 if parsing fails
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
    */
    
}

