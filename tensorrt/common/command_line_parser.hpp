
#ifndef COMMAND_LINE_PARSER_HPP_
#define COMMAND_LINE_PARSER_HPP_

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifdef __ANDROID__
#include <sstream>
namespace std
{

template <typename T> inline std::string to_string(const T &value)
{
        std::ostringstream os;
        os << value;
        return os.str();
}

} // namespace std
#endif

/**
 * @class CommandLineParser
 * @brief Parses command line's arguments and options
 */
class CommandLineParser
{
public:
        /**
         * @see Value class
         * @brief Argument or option's value type declaration
         */
        class Value;

        /**
         * @brief CommandLineParser constructor
         * @param[in] command_name Current command's name, default 'command'
         * @param[in] command_note Current command's note, default empty string
         */
        explicit CommandLineParser(const std::string &command_name = "command",
                                   const std::string &command_note = "");

        /**
         * @brief Adds one argument to parse from command line
         * @param[in] name Argument's name, only contains letter, number and '_'
         * @param[in] note Argument's note, default empty string
         * @param[in] only_number If argument's value only has number, default false
         * @return Returns true on success, returns false on failure
         */
        bool AddArgument(const std::string &name, const std::string &note = "",
                         bool only_number = false);

        /**
         * @brief Adds argument package to parse, contains at least one argument
         * @param[in] name Package's name, only contains letter, number and '_'
         * @param[in] note Package's note, default empty string
         * @param[in] only_number If package's values only have number, default false
         * @return Returns true on success, returns false on failure
         */
        bool AddArgumentPack(const std::string &name, const std::string &note = "",
                             bool only_number = false);

        /**
         * @brief Adds one option to parse, like a argument but with multi values
         * @param[in] name Option's name, like a argument's name but leading'--'
         * @param[in] value_count Option's value count, default 0
         * @param[in] short_name Option's short name, leading'-', default empty string
         * @param[in] note Option's note, default empty string
         * @param[in] only_number If option's value only has number, default false
         * @return Returns true on success, returns false on failure
         */
        bool AddOption(const std::string &name, size_t value_count = 0,
                       const std::string &short_name = "", const std::string &note = "",
                       bool only_number = false);

        /**
         * @brief Parses predefined arguments and options in command line
         * @param[in] argc Raw argument's count in command line, first is app path
         * @param[in] argv Raw argument's strings in command line
         * @return Returns true on success, returns false on failure
         */
        bool Parse(int argc, char *argv[]);

        /**
         * @brief Gets argument's value or option's values
         * @param[in] name Argument's name or option's name
         * @return Returns valid value on success, returns invalid value on failure
         */
        Value operator[](const std::string &name) const;

        /**
         * @brief Gets current command's usage info
         * @return Returns usage info on success, returns empty string on failure
         */
        std::string UsageInfo() const;

        /**
         * @brief Gets current command's help info
         * @return Returns help info on success, returns empty string on failure
         */
        std::string HelpInfo() const;

public:
        /**
         * @class Value
         * @brief Argument or option's value type, stored single or multiple value
         * @note Single value for argument, multiple value for option
         */
        class Value
        {
        public:
                friend class CommandLineParser;

                /**
                 * @brief Value default constructor
                 */
                Value()
                {
                        valid_ = false;
                };

                /**
                 * @brief Converts value or first value of multiple value to int
                 * @return Returns int value on success, throws an exception on failure
                 * @example if (parser["--count"]) { int count = parser["--count"]; }
                 */
                int ToInt() const
                {
                        return std::stoi(value_);
                }
                operator int() const
                {
                        return std::stoi(value_);
                }

                /**
                 * @brief Converts value or first value of multiple value to float
                 * @return Returns float value on success, throws an exception on failure
                 */
                float ToFloat() const
                {
                        return std::stof(value_);
                }
                operator float() const
                {
                        return std::stof(value_);
                }

                /**
                 * @brief Converts value or first value of multiple value to double
                 * @return Returns double value on success, throws an exception on failure
                 */
                double ToDouble() const
                {
                        return std::stod(value_);
                }
                operator double() const
                {
                        return std::stod(value_);
                }

                /**
                 * @brief Converts value or first value of multiple value to std::string
                 * @return Returns string value on success, throws an exception on failure
                 */
                std::string ToString() const
                {
                        return value_;
                }
                operator std::string() const
                {
                        return value_;
                }

                /**
                 * @brief Check if option is exists and parsed, or if value is available
                 * @return Returns true on success, returns false on failure
                 * @example if (parser["--count"]) { int count = parser["--count"]; }
                 */
                operator bool() const
                {
                        return valid_;
                }

                /**
                 * @brief Gets sub value count of multiple value (option's value count)
                 * @return Returns sub value count on success, returns 0 on failure
                 */
                size_t size()
                {
                        return values_.size();
                }

                /**
                 * @brief Gets sub value by index in multiple value
                 * @param[in] index Index of sub value
                 * @return Returns valid value on success, returns invalid value on failure
                 * @example for (size_t i = 0; i < parser["--option"].Count(); ++i) {
                 *            std::cout << parser["--option"][i] << std::endl;
                 *          }
                 */
                Value operator[](size_t index) const
                {
                        return index < values_.size() ? Value(values_[index]) : Value();
                }

        private:
                explicit Value(std::string str_value) : value_(std::move(str_value))
                {
                        valid_ = !value_.empty();
                }

                explicit Value(std::vector<std::string> str_values)
                        : values_(std::move(str_values))
                {
                        value_ = values_.empty() ? "" : values_[0];
                        valid_ = true;
                }

        private:
                bool valid_;
                std::string value_;
                std::vector<std::string> values_;
        };

private:
        bool VerifyName(const std::string &name)
        {
                if (name.empty() || !std::isalpha(name[0])) {
                        return false;
                }
                for (char c : name) {
                        if (!std::isalnum(c) && !std::isalpha(c) && c != '_') {
                                return false;
                        }
                }
                return true;
        }

private:
        struct Argument {
                std::string name;
                std::string note;
                bool only_number;
                bool is_package;
        };

        struct Option {
                std::string name;
                std::string short_name;
                std::string note;
                size_t value_count;
                bool only_number;
                size_t index;
        };

        mutable std::string app_name_;
        mutable std::string app_path_;
        std::string app_note_;

        bool have_argument_pack_ = false;
        std::vector<Argument> all_arguments_;
        std::map<std::string, Option> all_options_;
        std::map<std::string, std::string> short_name_to_name_of_option_;

        std::map<std::string, Value> cur_values_;
};

inline std::ostream &operator<<(std::ostream &cout, const CommandLineParser::Value &value)
{
        return cout << value.ToString();
}

inline CommandLineParser::CommandLineParser(const std::string &command_name,
                                            const std::string &command_note)
{
        app_name_ = command_name;
        app_note_ = command_note;
}

inline bool CommandLineParser::AddArgument(const std::string &name, const std::string &note,
                                           bool only_number)
{
        if (name.empty() || name.size() > 32 || !VerifyName(name)) {
                std::cerr << "AddArgumentFailed: name length must be <= 32 and start with "
                             "letter and contains only letter or number or '_'! name: "
                          << name << std::endl;
                return false;
        }
        for (auto &argument : all_arguments_) {
                if (name == argument.name) {
                        std::cerr
                            << "AddArgumentFailed: argument is already exist! name: " << name
                            << std::endl;
                        return false;
                }
        }
        all_arguments_.push_back({ name, note, only_number, false });
        return true;
}

inline bool CommandLineParser::AddArgumentPack(const std::string &name, const std::string &note,
                                               bool only_number)
{
        if (have_argument_pack_) {
                std::cerr << "AddArgumentPackFailed: already added one argument package! "
                             "just can add one argument package!"
                          << std::endl;
                return false;
        }
        if (name.empty() || name.size() > 32 || !VerifyName(name)) {
                std::cerr << "AddArgumentPackFailed: name length must be <= 32 and start with "
                             "letter and contains only letter or number or '_'! name: "
                          << name << std::endl;
                return false;
        }
        for (auto &argument : all_arguments_) {
                if (name == argument.name) {
                        std::cerr
                            << "AddArgumentPackFailed: name is already exist! name: " << name
                            << std::endl;
                        return false;
                }
        }
        all_arguments_.push_back({ name, note, only_number, true });
        have_argument_pack_ = true;
        return true;
}

inline bool CommandLineParser::AddOption(const std::string &name, size_t value_count,
                                         const std::string &short_name, const std::string &note,
                                         bool only_number)
{
        if (all_options_.count(name)) {
                std::cerr << "AddOptionFailed: option is already exist! name: " << name
                          << std::endl;
                return false;
        } else if (name.size() < 3 || name.size() > 32 || name[0] != '-' || name[1] != '-' ||
                   !VerifyName(name.substr(2))) {
                std::cerr << "AddOptionFailed: name length must be <= 32 and start with '--' and "
                             "contains only letter or number or '_'(except '--')! name: "
                          << name << std::endl;
                return false;
        } else if (!short_name.empty()) {
                if (short_name_to_name_of_option_.count(short_name)) {
                        std::cerr
                            << "AddOptionFailed: option short name is already exist! short name: "
                            << short_name << std::endl;
                        return false;
                } else if (short_name.size() < 2 || short_name.size() > 16 ||
                           short_name[0] != '-' || !VerifyName(short_name.substr(1))) {
                        std::cerr << "AddOptionFailed: short name length must be <= 16 and start "
                                     "with '-' and contains only letter or number or '_'(except "
                                     "'-')! short name: "
                                  << short_name << std::endl;
                        return false;
                }
        }
        all_options_[name].name = name;
        all_options_[name].note = note;
        all_options_[name].short_name = short_name;
        all_options_[name].only_number = only_number;
        all_options_[name].value_count = value_count;
        all_options_[name].index = all_options_.size() - 1;
        if (!short_name.empty()) {
                short_name_to_name_of_option_[short_name] = name;
        }
        return true;
}

inline bool CommandLineParser::Parse(int argc, char *argv[])
{
        auto show_usage_info_and_exit = [this]() {
                std::cerr << UsageInfo() << std::endl;
                if (!all_options_.count("--help")) {
                        std::cerr << "Try '" << (app_path_.empty() ? app_name_ : app_path_)
                                  << " --help' for more information." << std::endl;
                }
                exit(-1);
        };
        app_path_.clear();
        cur_values_.clear();
        if (argc < 1) {
                std::cerr << "ArgumentError: argument count must be > 0! current: " << argc
                          << std::endl;
                show_usage_info_and_exit();
        }
        app_path_ = argv[0];

        if (argc == 2 && !all_options_.count("--help") && std::string("--help") == argv[1]) {
                std::cout << HelpInfo() << std::endl;
                exit(0);
        }

        if (argc == 2 && !all_options_.count("--usage") && std::string("--usage") == argv[1]) {
                std::cout << UsageInfo() << std::endl;
                exit(0);
        }

        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i) {
                args.emplace_back(argv[i]);
        }

        std::vector<std::string> cur_arguments;
        std::map<std::string, std::vector<std::string>> cur_options;
        for (size_t i = 0; i < args.size(); ++i) {
                const auto &str = args[i];
                std::string option_name;
                if (all_options_.count(str)) {
                        option_name = str;
                } else if (short_name_to_name_of_option_.count(str)) {
                        option_name = short_name_to_name_of_option_[str];
                }

                if (option_name.empty()) {
                        cur_arguments.push_back(str);
                } else {
                        cur_options[option_name];
                        auto option_value_count = all_options_[option_name].value_count;
                        for (size_t j = 0; j < option_value_count; ++j) {
                                if (++i < args.size()) {
                                        if (all_options_[option_name].only_number) {
                                                try {
                                                        std::stoi(args[i]);
                                                } catch (std::exception &e) {
                                                        std::cerr << "ArgumentError: option ["
                                                                  << option_name
                                                                  << "] value require number!"
                                                                  << " current: " << args[i]
                                                                  << std::endl;
                                                        show_usage_info_and_exit();
                                                }
                                        }
                                        cur_options[option_name].push_back(args[i]);
                                } else {
                                        std::cerr
                                            << "ArgumentError: option [" << option_name
                                            << "] value count must be: " << option_value_count
                                            << ", current: " << j << std::endl;
                                        show_usage_info_and_exit();
                                }
                        }
                }
        }

        if (have_argument_pack_) {
                if (cur_arguments.size() < all_arguments_.size() - 1) {
                        std::cerr << "ArgumentError: argument count at least: "
                                  << all_arguments_.size() - 1
                                  << ", current: " << cur_arguments.size() << std::endl;
                        show_usage_info_and_exit();
                }
                auto pack_value_count = cur_arguments.size() - all_arguments_.size() + 1;
                for (size_t i = 0, j = 0; i < all_arguments_.size(); ++i, ++j) {
                        if (all_arguments_[i].is_package) {
                                if (pack_value_count <= 0) {
                                        std::cerr << "ArgumentError: argument pack <"
                                                  << all_arguments_[i].name
                                                  << "...> value count at least: 1"
                                                  << ", current: " << pack_value_count
                                                  << std::endl;
                                        show_usage_info_and_exit();
                                }
                                for (size_t k = 0; k < pack_value_count; ++k) {
                                        if (all_arguments_[i].only_number) {
                                                try {
                                                        std::stoi(cur_arguments[j]);
                                                } catch (std::exception &e) {
                                                        std::cerr
                                                            << "ArgumentError: argument pack <"
                                                            << all_arguments_[i].name
                                                            << "> value require number!"
                                                            << " current: " << cur_arguments[j]
                                                            << std::endl;
                                                        show_usage_info_and_exit();
                                                }
                                        }
                                        ++j;
                                }
                        } else {
                                if (all_arguments_[i].only_number) {
                                        try {
                                                std::stoi(cur_arguments[j]);
                                        } catch (std::exception &e) {
                                                std::cerr << "ArgumentError: argument <"
                                                          << all_arguments_[i].name
                                                          << "> value require number!"
                                                          << " current: " << cur_arguments[j]
                                                          << std::endl;
                                                show_usage_info_and_exit();
                                        }
                                }
                        }
                }
        } else {
                if (cur_arguments.size() != all_arguments_.size()) {
                        std::cerr
                            << "ArgumentError: argument count must be: " << all_arguments_.size()
                            << ", current: " << cur_arguments.size() << ", they are:";
                        auto max_count = std::max(cur_arguments.size(), all_arguments_.size());
                        for (size_t i = 0; i < max_count; ++i) {
                                std::cerr << " <";
                                std::cerr
                                    << (i < all_arguments_.size() ? all_arguments_[i].name : " ");
                                std::cerr << ":";
                                std::cerr << (i < cur_arguments.size() ? cur_arguments[i] : " ");
                                std::cerr << ">";
                        }
                        std::cerr << std::endl;
                        show_usage_info_and_exit();
                }
                for (size_t i = 0; i < all_arguments_.size(); ++i) {
                        if (all_arguments_[i].only_number) {
                                try {
                                        std::stoi(cur_arguments[i]);
                                } catch (std::exception &e) {
                                        std::cerr
                                            << "ArgumentError: argument <"
                                            << all_arguments_[i].name << "> value require number!"
                                            << " current: " << cur_arguments[i] << std::endl;
                                        show_usage_info_and_exit();
                                }
                        }
                }
        }

        if (have_argument_pack_) {
                auto pack_value_count = cur_arguments.size() - all_arguments_.size() + 1;
                for (size_t i = 0, j = 0; i < all_arguments_.size(); ++i) {
                        if (all_arguments_[i].is_package) {
                                std::vector<std::string> values;
                                for (size_t k = 0; k < pack_value_count; ++k) {
                                        values.push_back(cur_arguments[j]);
                                        ++j;
                                }
                                cur_values_[all_arguments_[i].name] = Value(values);
                        } else {
                                cur_values_[all_arguments_[i].name] = Value(cur_arguments[j]);
                                ++j;
                        }
                }
        } else {
                for (size_t i = 0; i < cur_arguments.size(); ++i) {
                        cur_values_[all_arguments_[i].name] = Value(cur_arguments[i]);
                }
        }

        for (const auto &option : cur_options) {
                cur_values_[option.first] = Value(option.second);
                auto short_name = all_options_[option.first].short_name;
                if (!short_name.empty()) {
                        cur_values_[short_name] = Value(option.second);
                }
        }
        return true;
}

inline CommandLineParser::Value CommandLineParser::operator[](const std::string &name) const
{
        return cur_values_.count(name) ? cur_values_.at(name) : Value();
}

inline std::string CommandLineParser::UsageInfo() const
{
        std::stringstream ss;
        ss << "Usage: " << (!app_path_.empty() ? app_path_ : "command");
        for (const auto &argument : all_arguments_) {
                ss << " <" << argument.name << (argument.is_package ? "..." : "")
                   << (argument.only_number ? ": NUM" : "") << ">";
        }
        std::vector<Option> all_options;
        for (const auto &option : all_options_) {
                all_options.push_back(option.second);
        }
        std::sort(all_options.begin(), all_options.end(),
                  [](const Option &op1, const Option &op2) { return op1.index < op2.index; });
        for (const auto &option : all_options) {
                ss << " [";
                ss << (option.short_name.empty() ? option.name
                                                 : option.short_name + "|" + option.name);
                for (size_t i = 0; i < option.value_count; ++i) {
                        ss << (option.only_number ? " N" : " V") << i + 1;
                }
                ss << "]";
        }
        return ss.str();
}

inline std::string CommandLineParser::HelpInfo() const
{
        std::stringstream ss;
        std::swap(app_name_, app_path_);
        ss << UsageInfo() << std::endl;
        std::swap(app_name_, app_path_);
        if (!app_note_.empty()) {
                ss << app_note_ << std::endl;
        }
        auto print_note = [&ss](std::string note, int left_width) {
                if (!note.empty()) {
                        size_t pos = std::string::npos;
                        while (!note.empty() && (pos = note.find('\n')) != std::string::npos) {
                                ss << " " << note.substr(0, pos);
                                note = note.substr(pos + 1);
                                ss << std::endl;
                                ss << std::left << std::setw(left_width) << " ";
                        }
                        if (!note.empty()) {
                                ss << " " << note;
                        }
                }
        };
        if (!all_arguments_.empty()) {
                ss << std::endl;
                ss << "Argument with '...' is package, 'N' means number, 'V' means string: "
                   << std::endl;
                int width = 0;
                std::vector<std::string> lefts;
                for (const auto &argument : all_arguments_) {
                        auto left = " <" + argument.name + (argument.is_package ? "...>" : ">") +
                                    (argument.only_number ? ": N" : ": V");
                        width = std::max(width, (int)left.size());
                        lefts.push_back(left);
                }
                width = std::min(50, width + 10);
                for (size_t i = 0; i < all_arguments_.size(); ++i) {
                        ss << std::left << std::setw(width) << lefts[i];
                        print_note(all_arguments_[i].note, width);
                        ss << std::endl;
                }
        }
        if (!all_options_.empty()) {
                ss << std::endl;
                ss << "Option value with 'N' means number, 'V' means string: " << std::endl;
                std::vector<Option> all_options;
                for (const auto &option : all_options_) {
                        all_options.push_back(option.second);
                }
                std::sort(
                    all_options.begin(), all_options.end(),
                    [](const Option &op1, const Option &op2) { return op1.index < op2.index; });
                int width = 0;
                std::vector<std::string> lefts;
                for (const auto &option : all_options) {
                        std::string left = " [";
                        if (option.short_name.empty()) {
                                left += option.name;
                        } else {
                                left += option.short_name + "|" + option.name;
                        }
                        if (option.value_count > 0) {
                                left += " ";
                        }
                        for (size_t i = 0; i < option.value_count; ++i) {
                                left += (option.only_number ? "N" : "V") + std::to_string(i + 1);
                                if (i < option.value_count - 1) {
                                        left += ' ';
                                }
                        }
                        left += "]";
                        width = std::max(width, (int)left.size());
                        lefts.push_back(left);
                }
                width = std::min(50, width + 10);
                for (size_t i = 0; i < all_options.size(); ++i) {
                        ss << std::left << std::setw(width) << lefts[i];
                        print_note(all_options[i].note, width);
                        ss << std::endl;
                }
        }
        return ss.str();
}

#endif // COMMAND_LINE_PARSER_HPP_