import './App.css'
import { useState , useEffect} from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale
} from 'chart.js';

ChartJS.register(
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale
);

function App() {
  const [formData, setFormData] = useState({
    BuildingType: '',
    PrimaryPropertyType: '',
    YearBuilt: '',
    NumberofBuildings: '',
    NumberofFloors: '',
    PropertyGFATotal: '',
    PropertyGFAParking: '',
    PropertyGFABuilding: '',
    SiteEnergyUse: '',
    Electricity: '',
    NaturalGas: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [showForm, setShowForm] = useState(true);
  const [historicalData, setHistoricalData] = useState(null);

  useEffect(() => {
    async function fetchHistoricalData() {
      try {
        const response = await axios.get('http://127.0.0.1:5000/get-historical-data');
        setHistoricalData(response.data);
      } catch (error) {
        console.error('Error fetching historical data:', error);
      }
    }
  
    fetchHistoricalData();
  }, []);
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };


  const handleSubmit = async (e) => {
    e.preventDefault();
    const numericData = {
      ...formData,
      YearBuilt: Number(formData.YearBuilt),
      NumberofBuildings: Number(formData.NumberofBuildings),
      NumberofFloors: Number(formData.NumberofFloors),
      PropertyGFATotal: Number(formData.PropertyGFATotal),
      PropertyGFAParking: Number(formData.PropertyGFAParking),
      PropertyGFABuilding: Number(formData.PropertyGFABuilding),
      SiteEnergyUse: Number(formData.SiteEnergyUse),
      Electricity: Number(formData.Electricity),
      NaturalGas: Number(formData.NaturalGas)
    };

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', numericData);
      console.log('Prediction:', response.data.prediction);
      setPrediction(response.data.prediction);
      setShowForm(false);
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };

  const chartData = {
    labels: historicalData ? historicalData['SiteEUI(kBtu/sf)'] : [],
    datasets: [
      {
        label: 'Property GFA Total',
        data: historicalData ? historicalData['PropertyGFATotal'] : [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'Electricity',
        data: historicalData ? historicalData['Electricity'] : [],
        borderColor: '#f87171',
        backgroundColor: 'rgba(248, 113, 113, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'Natural Gas',
        data: historicalData ? historicalData['NaturalGas'] : [],
        borderColor: '#34d399',
        backgroundColor: 'rgba(52, 211, 153, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'YearBuilt',
        data: historicalData ? historicalData['YearBuilt'] : [],
        borderColor: '#631f1f',
        backgroundColor: 'rgba(99, 31, 31,0.2)',
        borderWidth: 2,
      },
      {
        label: 'NumberofBuildings',
        data: historicalData ? historicalData['NumberofBuildings'] : [],
        borderColor: '#98a3a2',
        backgroundColor: 'rgba(152, 163, 162,0.2)',
        borderWidth: 2,
      },
      {
        label: 'NumberofFloors',
        data: historicalData ? historicalData['NumberofFloors'] : [],
        borderColor: '#36f000',
        backgroundColor: 'rgba(52, 255, 153, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'PrimaryPropertyType',
        data: historicalData ? historicalData['PrimaryPropertyType'] : [],
        borderColor: '#dbc51d',
        backgroundColor: 'rgba(219, 197, 29, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'SiteEnergyUse',
        data: historicalData ? historicalData['SiteEnergyUse'] : [],
        borderColor: '#3d4207',
        backgroundColor: 'rgba(61, 66, 7, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'BuildingType',
        data: historicalData ? historicalData['BuildingType'] : [],
        borderColor: '#d42f9d',
        backgroundColor: 'rgba(212, 47, 157, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'PropertyGFAParking',
        data: historicalData ? historicalData['PropertyGFAParking'] : [],
        borderColor: '#0f0b0e',
        backgroundColor: 'rgba(15, 11, 14, 0.2)',
        borderWidth: 2,
      },
      {
        label: 'PropertyGFABuilding',
        data: historicalData ? historicalData['PropertyGFABuilding'] : [],
        borderColor: '#a619bf',
        backgroundColor: 'rgba(166, 25, 191, 0.2)',
        borderWidth: 2,
      }
    ],
  };
  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: (context) => `Value: ${context.raw}`,
        },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Site EUI',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };
  
  return (
    <>
      <link
        rel="stylesheet"
        href="https://demos.creative-tim.com/notus-js/assets/styles/tailwind.css"
      />
      <link
        rel="stylesheet"
        href="https://demos.creative-tim.com/notus-js/assets/vendor/@fortawesome/fontawesome-free/css/all.min.css"
      />
      <section className="py-1 bg-blueGray-50">
        <div className="w-full lg:w-8/12 px-4 mx-auto mt-6">
          <div className="relative flex flex-col min-w-0 break-words w-full mb-6 shadow-lg rounded-lg bg-blueGray-100 border-0">
            <div className="rounded-t bg-white mb-0 px-6 py-6">
              <div className="text-center flex justify-between">
                <h6 className="text-blueGray-700 text-2xl font-bold">Site Energy Use Intensity AI Model</h6>
              </div>
            </div>
            <div className="flex-auto px-4 lg:px-10 py-10 pt-0">
            {showForm ? (
              <form onSubmit={handleSubmit}>
                <h6 className="text-blueGray-400 text-sm mt-3 mb-6 font-bold uppercase">
                  Building Details
                </h6>
                <div className="flex flex-wrap">
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="NumberofBuildings"
                      >
                        Number of Buildings
                      </label>
                      <input
                        type="number"
                        name="NumberofBuildings"
                        id="NumberofBuildings"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter number of buildings"
                        value={formData.NumberofBuildings}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="NumberofFloors"
                      >
                        Number of Floors
                      </label>
                      <input
                        type="number"
                        name="NumberofFloors"
                        id="NumberofFloors"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter number of floors"
                        value={formData.NumberofFloors}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="Electricity"
                      >
                        Electricity (kWh)
                      </label>
                      <input
                        type="number"
                        name="Electricity"
                        id="Electricity"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter electricity usage"
                        value={formData.Electricity}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="NaturalGas"
                      >
                        Natural Gas (therms)
                      </label>
                      <input
                        type="number"
                        name="NaturalGas"
                        id="NaturalGas"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter natural gas usage"
                        value={formData.NaturalGas}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <hr className="mt-6 border-b-1 border-blueGray-300" />
                <h6 className="text-blueGray-400 text-sm mt-3 mb-6 font-bold uppercase">
                  Property Details
                </h6>
                <div className="flex flex-wrap">
                <div className="w-full lg:w-6/12 px-4">
  <div className="relative w-full mb-3">
    <label
      className="block text-blueGray-600 text-sm font-bold mb-2"
    >
      Building Type
    </label>
    <select
      name="BuildingType"
      className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
      value={formData.BuildingType}
      onChange={handleChange}
    >
      <option value="">Select building type</option>
      <option value="NonResidential">NonResidential</option>
      <option value="SPS-District K-12">SPS-District K-12</option>
      <option value="Multifamily MR (5-9)">Multifamily MR (5-9)</option>
      <option value="Multifamily LR (1-4)">Multifamily LR (1-4)</option>
      <option value="Campus">Campus</option>
      <option value="Multifamily HR (10+)">Multifamily HR (10+)</option>
      <option value="Nonresidential COS">Nonresidential COS</option>
    </select>
  </div>
</div>

<div className="w-full lg:w-6/12 px-4">
  <div className="relative w-full mb-3">
    <label
      className="block text-blueGray-600 text-sm font-bold mb-2"
    >
      Primary Property Type
    </label>
    <select
      name="PrimaryPropertyType"
      className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
      value={formData.PrimaryPropertyType}
      onChange={handleChange}
    >
      <option value="">Select property type</option>
      <option value="Other">Other</option>
      <option value="Non-Refrigerated Warehouse">Non-Refrigerated Warehouse</option>
      <option value="Mixed Use Property">Mixed Use Property</option>
      <option value="Low-Rise Multifamily">Low-Rise Multifamily</option>
      <option value="Mid-Rise Multifamily">Mid-Rise Multifamily</option>
      <option value="High-Rise Multifamily">High-Rise Multifamily</option>
      <option value="K-12 School">K-12 School</option>
      <option value="Worship Facility">Worship Facility</option>
      <option value="Small- and Mid-Sized Office">Small- and Mid-Sized Office</option>
      <option value="College/University">College/University</option>
      <option value="Senior Care Community">Senior Care Community</option>
      <option value="Refrigerated Warehouse">Refrigerated Warehouse</option>
      <option value="Large Office">Large Office</option>
      <option value="Hotel">Hotel</option>
      <option value="Retail Store">Retail Store</option>
      <option value="Medical Office">Medical Office</option>
      <option value="Restaurant">Restaurant</option>
      <option value="Hospital">Hospital</option>
      <option value="Supermarket/Grocery Store">Supermarket/Grocery Store</option>
      <option value="Residence Hall/Dormitory">Residence Hall/Dormitory</option>
    </select>
  </div>
</div>

                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="YearBuilt"
                      >
                        Year Built
                      </label>
                      <input
                        type="number"
                        name="YearBuilt"
                        id="YearBuilt"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter year built"
                        value={formData.YearBuilt}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <hr className="mt-6 border-b-1 border-blueGray-300" />
                <h6 className="text-blueGray-400 text-sm mt-3 mb-6 font-bold uppercase">
                  Area Details
                </h6>
                <div className="flex flex-wrap">
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PropertyGFATotal"
                      >
                        Property GFA Total
                      </label>
                      <input
                        type="number"
                        name="PropertyGFATotal"
                        id="PropertyGFATotal"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter total GFA"
                        value={formData.PropertyGFATotal}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PropertyGFAParking"
                      >
                        Property GFA Parking
                      </label>
                      <input
                        type="number"
                        name="PropertyGFAParking"
                        id="PropertyGFAParking"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter GFA for parking"
                        value={formData.PropertyGFAParking}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="PropertyGFABuilding"
                      >
                        Property GFA Building
                      </label>
                      <input
                        type="number"
                        name="PropertyGFABuilding"
                        id="PropertyGFABuilding"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter GFA for building"
                        value={formData.PropertyGFABuilding}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                  <div className="w-full lg:w-6/12 px-4">
                    <div className="relative w-full mb-3">
                      <label
                        className="block text-blueGray-600 text-sm font-bold mb-2"
                        htmlFor="SiteEnergyUse"
                      >
                        Site Energy Use
                      </label>
                      <input
                        type="number"
                        name="SiteEnergyUse"
                        id="SiteEnergyUse"
                        className="border-0 px-3 py-3 placeholder-blueGray-300 text-blueGray-600 bg-white rounded text-sm shadow focus:outline-none focus:ring w-full ease-linear transition-all duration-150"
                        placeholder="Enter site energy use"
                        value={formData.SiteEnergyUse}
                        onChange={handleChange}
                      />
                    </div>
                  </div>
                </div>
                <div className="text-center mt-4">
                    <button
                      type="submit"
                      className="bg-blue-500 text-white
                       active:bg-blue-600 font-bold uppercase text-sm px-6 py-3 rounded shadow hover:shadow-lg outline-none focus:outline-none ease-linear transition-all duration-150"
                    >
                      Get Prediction
                    </button>
                  </div>
              </form>
            ) : (
              <>
              <div className="text-center py-10">
                <div className="bg-white shadow-lg rounded-lg p-6">
                  <h3 className="text-2xl font-bold text-blueGray-700 mb-4">
                    Prediction Result
                  </h3>
                  <p className="text-blueGray-600 text-lg">
                    The predicted Site Energy Use Intensity (EUI) is:
                  </p>
                  <div className="mt-4 p-4 bg-blueGray-100 rounded">
                    <h2 className="text-xl font-semibold text-blueGray-700">
                      {prediction}  kBtu/sf
                    </h2>
                  </div>
                  <button onClick={() => setShowForm(true)} className="bg-blue-500 mt-6 text-white px-4 py-2 rounded">
            Make Another Prediction
          </button>
                </div>
              </div>
              {historicalData && (
        <div>
          <h3 className="text-center text-xl font-bold mb-4">Historical Data</h3>
          <Line data={chartData} options={options} />
        </div>
      )}
              </>
            )}

    </div>
            </div>
          </div>
      </section>
      </>
  );
}

export default App;