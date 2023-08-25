using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LapTimeManager : MonoBehaviour
{
    public static List<float> lapTimes = new List<float>();

    public static void AddLapTime(float time)
    {
        lapTimes.Add(time);
  
    }

    public static void ResetLapTimes()
    {
        lapTimes.Clear();
    }
}
