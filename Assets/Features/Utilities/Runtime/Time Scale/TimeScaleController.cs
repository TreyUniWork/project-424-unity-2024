using UnityEngine;
using System;
namespace Perrinn424.Utilities
{
    public class TimeScaleController : MonoBehaviour
    {
        // AUT - Eirik
        public float defaultRealtimeValue = 1.0f;

        [SerializeField]
        private float[] timeScales;
        private ClampedIterator<float> timeScalesIterator;
        private int realTimeIndex;

        public float TimeScale => timeScalesIterator.Current;
        public bool IsRealTime => timeScalesIterator.CurrentIndex == realTimeIndex;

        public event Action<float> onTimeScaleChanged;

        private void OnEnable()
        {
            CreateIterator();
        }

        private void CreateIterator()
        {
            if (!IsSorted(timeScales))
            {
                throw new ArgumentException($"{nameof(timeScales)} must be sorted");
            }

            realTimeIndex = Array.FindIndex(timeScales, e => e == defaultRealtimeValue); // AUT - changed form 1.0f to defaultRealtimeValue
            if (realTimeIndex == -1)
            {
                throw new ArgumentException($"{nameof(timeScales)} must contains at least one value = 1.0f (real time)");
            }

            timeScalesIterator = new ClampedIterator<float>(timeScales, realTimeIndex);
            UpdateTimeScale();
        }

        internal void NextTimeScale()
        {
            timeScalesIterator.MoveNext();
            UpdateTimeScale();
        }

        internal void PreviousTimeScale()
        {
            timeScalesIterator.MovePrevious();
            UpdateTimeScale();
        }

        private void UpdateTimeScale()
        {
            Time.timeScale = timeScalesIterator.Current;
            onTimeScaleChanged?.Invoke(timeScalesIterator.Current);
        }

        private bool IsSorted(float[] arr)
        {
            for (int i = 1; i < arr.Length; i++)
            {
                if (arr[i - 1] > arr[i])
                {
                    return false;
                }
            }
            return true;
        }
    }
}
