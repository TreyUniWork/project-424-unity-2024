using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Perrinn424.AutopilotSystem;
using VehiclePhysics.Timing;
using System.IO;
using System.Linq;
using UnityEngine.UI;
using UnityEditor;
using System;


public class SimulationManager : MonoBehaviour
{
    // for asset file stuff
    public Autopilot autopilot;
    private int maxAssetsPerGeneration = 5;
    public int currentAssetIndex { get; private set; } = 0;
    public int carNumber { get; private set; } = 0;
    public int currentGenerationNumber { get; private set; } = -1; // different from index because the way the folders are sorted
    public int currentGenerationIndex { get; private set; } = -1; // Start with -1 to indicate no generation selected yet
    private string basePath = "Assets/Resources/GeneticAssets";
    private const string CounterKey = "CurrentAssetIndex";

    // for ui
    [SerializeField] public Text genNumberText;
    [SerializeField] public Text carNumberText;

    private void Start()
    {
        // init asset file
        currentAssetIndex = PlayerPrefs.GetInt(CounterKey, 0);

        if (currentGenerationIndex == 0 && currentAssetIndex == 0)
        {
            SelectLatestGeneration();
        }
        else
        {
            SwitchOutLap();
        }
    }

    private void Update()
    {
        // update ui
        genNumberText.text = (currentGenerationNumber) + "";
        carNumberText.text = currentAssetIndex + "";
    }

    public void SwitchOutLap()
    {
        SelectLatestGeneration();
    }


    // selects the GEN* folder with the highest number
    private void SelectLatestGeneration()
    {
        int maxGeneration = -1;
        string latestGenerationFolder = null;

        string[] folderPaths = Directory.GetDirectories(basePath);
        currentGenerationNumber = folderPaths.Length;

        foreach (string folderPath in folderPaths)
        {
            string folderName = System.IO.Path.GetFileName(folderPath);

            // Check if the folder name starts with "GEN" and the rest is a number
            if (folderName.StartsWith("GEN") && int.TryParse(folderName.Substring(3), out int generation))
            {
                if (generation > maxGeneration)
                {
                    maxGeneration = generation;
                    latestGenerationFolder = folderPath;
                }
            }
        }

        if (!string.IsNullOrEmpty(latestGenerationFolder))
        {
            currentGenerationIndex = Array.IndexOf(folderPaths, latestGenerationFolder);
            CycleToNextAsset();
        }
        else
        {
            Debug.Log("No generation folders found.");
        }
    }


    // selects the next asset in the folder
    private void CycleToNextAsset()
    {
        string[] folderPaths = Directory.GetDirectories(basePath, "GEN*");

        if (folderPaths.Length > 0 && currentGenerationIndex >= 0 && currentGenerationIndex < folderPaths.Length)
        {
            currentAssetIndex = (currentAssetIndex % maxAssetsPerGeneration) + 1;

            // Increment carNumber and wrap it if it goes above 5
            carNumber = (carNumber % maxAssetsPerGeneration) + 1;

            string currentGenerationFolder = folderPaths[currentGenerationIndex];
            string currentAssetFile = System.IO.Path.Combine(currentGenerationFolder, $"gen{currentGenerationNumber}asset{currentAssetIndex}.asset");

            if (File.Exists(currentAssetFile))
            {
                // refresh assetdatabase to get meta file
                AssetDatabase.Refresh();

                Debug.Log("Selected File: " + currentAssetFile);
                // Load lap
                LoadLap(currentAssetFile);

                // Save currentAssetIndex externally so when it the scene reloads, it remembers.
                PlayerPrefs.SetInt(CounterKey, currentAssetIndex);
            }
            else
            {
                Debug.LogWarning($"Asset file not found: {currentAssetFile}");
            }
        }
        else
        {
            Debug.Log("Invalid generation index or no generation folders found.");
        }
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

    // switch out the lap in unity.
    private void LoadLap(string assetPath)
    {
        // trim asset path so it doesn't include resources (very hacky)
        string resourcesPath = "Assets/Resources/";

        if (assetPath.StartsWith(resourcesPath))
        {
            int startIndex = assetPath.IndexOf(resourcesPath) + resourcesPath.Length;
            assetPath = assetPath.Substring(startIndex);
            int dotIndex = assetPath.LastIndexOf('.');

            if (dotIndex >= 0)
            {
                assetPath = assetPath.Substring(0, dotIndex);
            }
        }

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