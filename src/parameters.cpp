#include <dawn/parameters.hxx>

void DAWN::IO::printHelp() {
  const int optionWidth = 30;
  std::cout << "Usage: program [options]\n"
            << "Options:\n"
            << std::left  // 左对齐
            << "  " << std::setw(optionWidth) << "-h, --help"
            << "Show this help message and exit\n"
            << "  " << std::setw(optionWidth) << "-i, --input PATH"
            << "Input file path\n"
            << "  " << std::setw(optionWidth) << "-o, --output PATH"
            << "Output file path\n"
            << "  " << std::setw(optionWidth) << "-l, --sourceList PATH"
            << "List of Source nodes file path\n"
            << "  " << std::setw(optionWidth) << "-p, --print [true|false]"
            << "Print source information\n"
            << "  " << std::setw(optionWidth) << "-w, --weighted [true|false]"
            << "Set graph as weighted\n"
            << "  " << std::setw(optionWidth) << "-s, --source NUM"
            << "Source node (optional, defaults to a random value)\n";
}

DAWN::IO::parameters_t DAWN::IO::parameters(int argc, char* argv[]) {
  DAWN::IO::parameters_t params;
  int opt;
  int option_index = 0;

  struct option long_options[] = {
      {"help", no_argument, nullptr, 'h'},
      {"input", required_argument, nullptr, 'i'},
      {"output", required_argument, nullptr, 'o'},
      {"sourceList", required_argument, nullptr, 'l'},
      {"print", required_argument, nullptr, 'p'},
      {"weighted", required_argument, nullptr, 'w'},
      {"source", required_argument, nullptr, 's'},
      {0, 0, 0, 0}};

  while ((opt = getopt_long(argc, argv, "hi:o:l:p:w:s:", long_options,
                            &option_index)) != -1) {
    switch (opt) {
      case 'h':
        DAWN::IO::printHelp();
        exit(EXIT_SUCCESS);
      case 'i':
        params.input_path = optarg;
        break;
      case 'o':
        params.output_path = optarg;
        break;
      case 'l':
        params.sourceList_path = optarg;
        break;
      case 'p':
        if (std::string(optarg) == "true") {
          params.print = true;
        } else if (std::string(optarg) == "false") {
          params.print = false;
        } else {
          std::cerr
              << "Invalid value for print option. Use 'true' or 'false'.\n";
          exit(EXIT_FAILURE);
        }
        break;
      case 'w':
        if (std::string(optarg) == "true") {
          params.weighted = true;
        } else if (std::string(optarg) == "false") {
          params.weighted = false;
        } else {
          std::cerr
              << "Invalid value for weighted option. Use 'true' or 'false'.\n";
          exit(EXIT_FAILURE);
        }
        break;
      case 's':
        params.source = std::atoi(optarg);
        break;
      default:
        DAWN::IO::printHelp();
        exit(EXIT_FAILURE);
    }
  }

  // Check for required arguments
  if (params.input_path.empty()) {
    std::cerr << "Error: Missing the input file path.\n";
    DAWN::IO::printHelp();
    exit(EXIT_FAILURE);
  }

  // If source is not provided, generate a random value
  if (params.source == -1) {
    params.source = std::rand();
  }

  return params;
}