﻿using System.IO;
using EdyCommonTools.EditorTools;
using UnityEngine;
using UnityEditor;
using VehiclePhysics;

namespace Perrinn424.UI.Editor
{
    public class ReplayImageGenerator : EditorWindow
    {
        private VPReplayAsset replay;
        public RepalyTexture replayTexture;

        public float rate = 0.02f;
        public int resolution = 400;
        private UnityEditor.Editor replayEditor;

        [MenuItem("Tools/Vehicle Physics/Replay Image Generator")]
        static void Init()
        {
            // Get existing open window or if none, make a new one:
            ReplayImageGenerator window = (ReplayImageGenerator)GetWindow(typeof(ReplayImageGenerator));
            window.Show();
        }

        void OnGUI()
        {
            replay = EditorGUILayout.ObjectField(replay, typeof(VPReplayAsset), false) as VPReplayAsset;

            if (replay == null)
                return;

            DrawReplayData();

            DrawSettings();

            DrawRefresh();

            DrawReplayImage();

            DrawSave();
        }

        private void DrawReplayData()
        {
            GUI.enabled = false;
            replayEditor = UnityEditor.Editor.CreateEditor(replay);
            replayEditor.OnInspectorGUI();
            GUI.enabled = true;
        }

        private void DrawSettings()
        {
            EditorGUILayout.BeginHorizontal();
            rate = EditorGUILayout.FloatField("Rate", rate);
            EditorGUILayout.LabelField($"{rate:F2} s", $"{1f / rate:F2} Hz");
            EditorGUILayout.EndHorizontal();

            if (rate < replay.timeStep)
            {
                EditorGUILayout.HelpBox("Image rate sampling should be less than replay rate sampling", MessageType.Warning);
            }
            resolution = EditorGUILayout.IntField("Resolution", resolution);
        }

        private void DrawRefresh()
        {
            if (GUILayout.Button("Refresh") || replayTexture == null)
            {
                replayTexture = new RepalyTexture(resolution, replay, rate);
            }
        }

        private void DrawReplayImage()
        {
            if (replayTexture != null)
            {
                EditorGUILayout.LabelField("Frames", replayTexture.SamplingCount.ToString());
                Rect graphRect = EditorGUILayout.GetControlRect(false, replayTexture.Resolution);
                TextureCanvasEditor.InspectorDraw(replayTexture.canvas, graphRect);
            }
        }

        private void DrawSave()
        {
            if (GUILayout.Button("Save as Image"))
            {
                string dirPath = "Assets/Replay Images/";

                var imagePath = EditorUtility.SaveFilePanelInProject("Save Replay Image", replay.name + ".png", "png", "", dirPath);

                if (imagePath.Length != 0)
                {
                    var pngData = replayTexture.canvas.texture.EncodeToPNG();
                    if (pngData != null)
                    {
                        SaveAndImport(imagePath, pngData);
                        SelectInProject(imagePath);
                    }
                }
            }
        }

        private static void SaveAndImport(string imagePath, byte[] pngData)
        {
            File.WriteAllBytes(imagePath, pngData);
            AssetDatabase.ImportAsset(imagePath);
            TextureImporter textureImporter = AssetImporter.GetAtPath(imagePath) as TextureImporter;
            textureImporter.textureType = TextureImporterType.Sprite;
            textureImporter.SaveAndReimport();
        }

        private static void SelectInProject(string imagePath)
        {
            var obj = AssetDatabase.LoadAssetAtPath(imagePath, typeof(Texture2D));
            Selection.activeObject = obj;
            EditorUtility.FocusProjectWindow();
        }
    } 
}