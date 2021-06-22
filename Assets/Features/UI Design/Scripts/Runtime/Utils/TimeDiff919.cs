﻿using VehiclePhysics;

namespace Perrinn424
{
    public class TimeDiff919 : VehicleBehaviour
    {
        private TimeReference porsche;
        private TimeReference volkswagen;

        public float PorscheDiff { get; private set; }
        public float VolkswagenDiff { get; private set; }

        public float PorscheSpeed { get; private set; }
        public float VolkswagenSpeed { get; private set; }

        public override void OnEnableVehicle()
        {
            porsche = TimeReferenceHelper.CreatePorsche();
            volkswagen = TimeReferenceHelper.CreateVolkswagen();
        }

        private void Update()
        {
            Telemetry.DataRow latestTelemetry = vehicle.telemetry.latest;
            Update((float)latestTelemetry.time, (float)latestTelemetry.distance);
        }

        private void Update(float currentTime, float currentDistance)
        {
            PorscheDiff = porsche.LapDiff(currentTime, currentDistance);
            VolkswagenDiff = volkswagen.LapDiff(currentTime, currentDistance);

            PorscheSpeed = porsche.Speed;
            VolkswagenSpeed = volkswagen.Speed;
        }
    }
}
