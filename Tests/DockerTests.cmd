% WSL Example usage:

% sudo mkdir -p /mnt/data
% sudo mount -t drvfs L:/Archief/active/cellular_imaging/OMERO_test/ValidateDocker /mnt/data

% RGB: LIF/LOF/XLEF test images:
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/RGB.lif --image_uuid 710afbc4-24d7-11f0-bebf-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/RGB.lif --image_uuid b98ee309-6d42-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/LargeImage.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/SmallImage.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/RGB-XLEF/RGB.xlef --image_uuid 710afbc4-24d7-11f0-bebf-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/RGB-XLEF/RGB.xlef --image_uuid b98ee309-6d42-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress

% MultiChannel: LIF/LOF/XLEF test images:
% TileScan_Large_Merged
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel.lif --image_uuid cc1967e9-6de7-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/TileScan_Large_Merged.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel/MultiChannel.xlef --image_uuid cc1967e9-6de7-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile "/data/Slide 3-Mosaic001_ICC_Merged.lof" --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% TileScan_Small_Merged
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel.lif --image_uuid 00d5c2c9-6de8-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/TileScan_Small_Merged.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel/MultiChannel.xlef --image_uuid 00d5c2c9-6de8-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% TimeSeries_Large
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel.lif --image_uuid eaa827f1-6de8-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/TimeSeries_Large.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel/MultiChannel.xlef --image_uuid eaa827f1-6de8-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% TimeSeries_Small
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel.lif --image_uuid cebbad95-6de8-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/TimeSeries_Small.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel/MultiChannel.xlef --image_uuid cebbad95-6de8-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% ZStack_Large
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel.lif --image_uuid 44b21ebf-6de9-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/ZStack_Large.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel/MultiChannel.xlef --image_uuid 44b21ebf-6de9-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% ZStack_Small
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel.lif --image_uuid 5be4452b-6de9-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/ZStack_Small.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/MultiChannel/MultiChannel.xlef --image_uuid 5be4452b-6de9-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress


% MultiChannel: Special Case Negative Overlap LIF/LOF/XLEF test images:
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/NegOverlapTilescan-2t-3pos.lif --image_uuid 7dcbf9b7-6de2-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/NegOverlapTilescan-2t-3pos.lof --outputfolder "/out" --altoutputfolder "/outalt" --show_progress
% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/NegOverlapTilescan-2t-3pos/NegOverlapTilescan-2t-3pos.xlef --image_uuid 7dcbf9b7-6de2-11f0-bed3-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress

% docker run --rm -v "/mnt/data":/data -v "/mnt/data/.processed":/out -v "/mnt/data/out":/outalt convertleica-docker --inputfile /data/RGB-XLEF-Multilevel/RGB-Multilevel.xlef --image_uuid 84940db9-8403-11f0-bed5-80e82ce1e716 --outputfolder "/out" --altoutputfolder "/outalt" --show_progress